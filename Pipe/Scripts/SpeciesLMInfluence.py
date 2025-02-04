import os
import glob
import yaml
import argparse
import itertools
import math
import contextlib

import numpy as np
import pandas as pd
import tqdm
import torch 

from transformers import AutoModelForSequenceClassification, DefaultDataCollator
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertConfig
from flash_attn.models.bert import BertModel, BertForPreTraining
from flash_attn.models.bert import remap_state_dict
from flash_attn.utils.pretrained import state_dict_from_pretrained
from transformers import BertConfig
from datasets import Dataset

import scipy
import torch.nn as nn
from torch.amp import autocast
import anndata as ad

parser = argparse.ArgumentParser()
parser.add_argument('-in_path', type=str, required="True")
parser.add_argument('-out_path', type=str, required="True")
parser.add_argument('-config_path', type=str, required="True")
args = parser.parse_args()

in_path = args.in_path
out_path = args.out_path
config_path = args.config_path

# load and chunk
# parse out indices
idx = out_path.split(".")[-2].split("_")[-2:]
start_idx = int(idx[0])
end_idx = int(idx[1])
print(start_idx, end_idx)
bed_cols = list(pd.read_table(in_path, nrows=0).columns)
table = pd.read_table(in_path,skiprows=start_idx+1, nrows=(end_idx+1)-start_idx, names=bed_cols)
bed_cols_req = ['Chromosome','Start','End','Ref','Alt','seq','variant','rel_pos']
bed_cols = [x for x in bed_cols_req if x in table.columns]
assert all(x == y for x,y in zip(bed_cols,bed_cols_req))
table = table[bed_cols]

# ## PARAMS

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

proxy_species = config["proxy_species"]#"homo_sapiens"
model_path = config["model_path"]#'/s/project/denovo-prosit/JohannesHingerl/BERTADN/outputs_BERTADN_flashed_metazoa_upstream2000/checkpoint-200000'
k = config['k']#6
left_special_tokens = config['left_special_tokens']#2

# ## CODE

def kmers_stride1(seq, k=1):
    # splits a sequence into overlapping k-mers
    return [seq[i:i + k] for i in range(0, len(seq)-k+1)]  

def tok_func_species(x, species_proxy, seq_col, tokenizer, k=6):
    res = tokenizer(species_proxy + " " +  " ".join(kmers_stride1(x[seq_col], k=k)))
    return res

def load_lm_model(model_path, disable_optimizations=False):
    print ('Attempting loading flashed model.')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path)    
    #print(config)
    flashed = True
    model = BertForPreTraining.from_pretrained(model_path, config)
    return model, tokenizer

class ReconstructionModelAllPred(nn.Module):
    
    def __init__(self, lm, tokenizer, device,
                 kmer_size = 6, 
                 left_special_tokens = 2,
                 right_special_tokens = 1,
                 mask_token=4,
                 only_predict_masked = False,
                 num_special_token_types = 5,
                 float16=True,
                 require_grad=False,
                 logits_key="prediction_logits",
                ):
        super().__init__()
        self.lm = lm
        self.tokenizer = tokenizer
        self.require_grad = require_grad
        self.only_predict_masked = only_predict_masked
        self.kmer_size = kmer_size
        self.device = device
        self.float16 = float16
        self.left_special_tokens = left_special_tokens
        self.right_special_tokens = right_special_tokens
        self.num_special_token_types = num_special_token_types
        self.mask_token = mask_token
        self.create_prb_filter()
        self.logits_key = logits_key
        
        self.word_embeddings = None
        self.hook_dict = {}
        
    def set_grad_computation(self, switch):
        self.require_grad = switch
        
    def set_word_embedding_hook(self):
        def getHook():
            def hook(model, input, output):
                output.retain_grad()
                self.word_embeddings = output
            return hook
        self.hook_dict["words"] = self.lm.bert.embeddings.word_embeddings.register_forward_hook(getHook())
        
    def create_prb_filter(self):
        """make a convolutional filter for each nt
        The way this works:
        Take the kmer ACGTGC which maps to token 739, its last nt is C
        This would be the prediction for the masked nucleotide from this kmer, if the kmer is the first in a masked span
        So the first row of column 739 searches for C, in other words filter_xyz = 1 for x = 0, y = 739, z = 2
        Equally, the second row of column 739 searches for G etc..."""
        vocab = self.tokenizer.get_vocab()
        kmer_list = ["".join(x) for x in itertools.product("ACGT",repeat=self.kmer_size)]
        nt_mapping = {"A":0,"C":1,"G":2,"T":3}
        prb_filter = np.zeros((self.kmer_size, 4**self.kmer_size, 4))
        for kmer in kmer_list:
            token = vocab[kmer] - self.num_special_token_types # there are 5 special tokens
            for idx, nt in enumerate(kmer):
                nt_idx = nt_mapping[nt]
                prb_filter[(self.kmer_size-1)-idx, token, nt_idx] = 1
        prb_filter = torch.from_numpy(prb_filter)
        self.prb_filter = prb_filter.to(self.device) # k, 4**k, 4
        if self.float16:
            self.prb_filter = self.prb_filter.to(torch.bfloat16)
        else:
            self.prb_filter = self.prb_filter.float() 
    
    #@autocast(device)
    def forward(self, tokens):
        autocast_context = contextlib.nullcontext() if self.float16 else autocast(self.device) 
        grad_context = contextlib.nullcontext() if self.require_grad else torch.no_grad() 
        with autocast_context:
            with grad_context:
                predictions = self.lm(tokens)[self.logits_key] # Dim: B, L_in, vocab_size
                logits = predictions[:,:,self.num_special_token_types:(self.num_special_token_types+self.prb_filter.shape[1])] # remove any non k-mer dims (there are 5 special tokens)
                kmer_preds = torch.softmax(logits,dim=2)
                # remove special tokens:
                kmer_preds = kmer_preds[:,(self.left_special_tokens):(kmer_preds.shape[1] - self.right_special_tokens),:]
                # pad to predict first k-1 and last k-1 nt
                if self.kmer_size > 1:
                    kmer_pad = torch.zeros((kmer_preds.shape[0], (self.kmer_size-1), kmer_preds.shape[2]),device=self.device)
                    kmer_preds = torch.concat([kmer_pad,kmer_preds,kmer_pad],axis=1)
                # reshape so that each span (representing one nucleotide) is its own entry
                kmer_preds = kmer_preds.unfold(dimension=1,size=self.kmer_size,step=1).swapaxes(2,3) # B, L_seq, k, 4**k
                # convert kmer predictions to nucleotide predictions
                nt_preds = kmer_preds.unsqueeze(-1).expand((kmer_preds.shape[0],kmer_preds.shape[1],kmer_preds.shape[2],kmer_preds.shape[3],4)) # B, L_seq, k, 4**k, 4
                nt_preds = (nt_preds * self.prb_filter).sum(axis=(2,3)) # B, L_seq, 4
                # renormalize so that it sums to one
                nt_prbs = nt_preds/self.kmer_size
                if self.only_predict_masked: # only record predictions for properly masked spans
                    # find the properly masked spans
                    # remove special tokens
                    tokens = tokens[:,self.left_special_tokens:(tokens.shape[1] - self.right_special_tokens)]
                    # pad with mask token (to ensure edge cases become spans)
                    token_pad = torch.zeros((tokens.shape[0],self.kmer_size-1), dtype=torch.int64, device=self.device) + 4
                    tokens = torch.concat([token_pad,tokens,token_pad],axis=1)
                    # unfold to B, L_seq, k
                    tokens = tokens.unfold(dimension=1,size=self.kmer_size,step=1)
                    # find masked spans
                    masked_positions = ((tokens == self.mask_token).sum(axis=2) == self.kmer_size)
                    nt_prbs = nt_prbs * masked_positions.unsqueeze(-1) # mask
        return nt_prbs
    
    def predict_all_from_dataloader(self, data_loader, batch_size = None):
        assert not self.only_predict_masked
        output_arrays = []
        for i, batch in tqdm.tqdm(enumerate(data_loader)):
            # get some tokenized sequences (B, L_in)
            tokens = batch['input_ids']
            # predict
            outputs = self(tokens.to(self.device)) # B, L_seq, 4
            output_arrays.append(outputs.cpu()) # send to cpu to conserve memory
        # rebuild to B, L_seq, 4
        predictions = torch.concat(output_arrays, axis=0)
        return predictions

class VariantDataset(torch.utils.data.Dataset):
    def __init__(self, variant_df, tok_func, kmer_size=6, left_special_tokens=2, right_special_tokens=1, mask_token=4):
        self.variant_df = variant_df
        self.tok_func = tok_func
        self.kmer_size = kmer_size
        self.left_special_tokens = left_special_tokens
        self.right_special_tokens = right_special_tokens
        self.mask_token = mask_token

    def __getitem__(self, index):
        row = self.variant_df.iloc[index]
        # get position
        pos = row["rel_pos"]
        # get sequence
        seq = row["seq"]
        assert row["seq"][pos] == row["Ref"]
        # tokenize sequence
        tok = torch.tensor(self.tok_func(seq)["input_ids"])
        # insert variant
        seq_alt = row['seq'][:pos] + row['Alt'] + row['seq'][pos+1:]
        tok_alt = torch.tensor(self.tok_func(seq_alt)["input_ids"])
        return torch.stack([tok, tok_alt]), torch.tensor(pos)
        
    def __len__(self):
        return len(self.variant_df)

### Prepare model

model, tokenizer = load_lm_model(model_path)

tok_func = lambda x: tokenizer(proxy_species + " " +  " ".join(kmers_stride1(x, k=k)))

device = "cuda"
model.to(torch.bfloat16)#.to(device)
model.to(device)
model.eval()
print("flashed")
print ("Done.")

reconstructor = ReconstructionModelAllPred(model, tokenizer, device, float16=True, only_predict_masked=False, kmer_size=k, left_special_tokens=left_special_tokens)

### Run

ds = VariantDataset(table, tok_func)
dl = torch.utils.data.dataloader.DataLoader(ds, num_workers=8, pin_memory=True, batch_size=24)

influence = []

eps = 1e-4

for batch in tqdm.tqdm(dl):
    tokens, positions = batch
    bs = tokens.shape[0]
    tokens = tokens.reshape(bs*2,tokens.shape[2])
    prbs = reconstructor(tokens.to(device)).float()
    prbs = prbs.reshape(bs, 2, -1, 4)
    # renormalize in float
    prbs = prbs + eps
    prbs = prbs/prbs.sum(axis=-1,keepdim=True)
    # compute influence score
    odds = torch.log(prbs) - torch.log(1 - prbs)
    influence_score = (torch.abs(odds[:,1] - odds[:,0])).amax(axis=-1)
    influence_score[torch.arange(influence_score.shape[0]),positions] = 0 # set self interaction to zero
    influence_score = influence_score.mean(axis=-1)
    influence.append(influence_score.cpu())

influence = torch.concat(influence,axis=0)

adata = ad.AnnData(influence.unsqueeze(-1).numpy(), 
                   obs=table,
                   var=pd.DataFrame({'Score':['Influence']}),
                  )

adata.write(out_path, compression="gzip")