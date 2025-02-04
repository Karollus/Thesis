import os
import yaml
import argparse

import numpy as np
import pandas as pd
import polars as pl
import anndata as ad
import scipy
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from enformer_pytorch.data import GenomeIntervalDataset, str_to_one_hot
from scooby.utils.transcriptome import Transcriptome
from scooby.utils.utils import undo_squashed_scale, fix_rev_comp_multiome, get_pseudobulk_count_pred, get_gene_slice_and_strand
from scooby.modeling import Scooby

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
print("Load data")
names = list(pd.read_csv(in_path, nrows=2, sep="\t").columns)
dataset = pd.read_csv(in_path, skiprows=start_idx+1, nrows=(end_idx+1)-start_idx, names=names, sep="\t")
print("Loaded data")
print(dataset.iloc[0])

# load config
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# get config params
targets_path = config['targets_path']#'/s/project/QNA/borzoi_training_data/hg38/targets.txt'
data_path = config['data_path']#'/data/ceph/hdd/project/node_08/QNA/scborzoi/neurips_bone_marrow/'
gtf_file = config['gtf_file']#"/data/ceph/hdd/project/node_08/QNA/scborzoi/neurips_bone_marrow/gencode.v32.annotation.sorted.gtf.gz"
fasta_file = config['fasta_file']
bed_file = config['bed_file']
pretrained_path = config['pretrained_path']
embedding_path = config['embedding_path']#os.path.join(data_path, 'borzoi_training_data_fixed',  'embedding_no_val_genes_new.pq')
cell_type_idx_path = config['cell_type_idx_path']#os.path.join(data_path,  'borzoi_training_data_fixed/celltype_fixed.pq')

print(pretrained_path)
print(config_path)

# Load data
borzoi_targets = pd.read_table(targets_path).rename(columns={'Unnamed: 0':'index'})
rna_tracks = borzoi_targets.loc[borzoi_targets.description.str.startswith('RNA')].index

# VariantDataSet
class VariantDataset(Dataset):

    def __init__(self, snp_df, gtf_file, fasta_file, bed_file,
                context_length = 524288):
        # load gene regions
        gene_regions = pd.read_table(bed_file,names=['Chromosome','Start','End','gene_name','Strand'])
        # extend gene regions
        gene_regions['Start'] = gene_regions['Start'] - 163840
        gene_regions['End'] = gene_regions['End'] + 163840
        # intersect variants with gene regions (of target gene)
        snp_df = gene_regions.merge(snp_df,on=['Chromosome','gene_name'])
        self.snp_df = snp_df.query('Pos >= Start and Pos < End').reset_index(drop=True)
        # get a way to query the gene regions
        self.gene_to_idx = {g:i for i,g in enumerate(gene_regions['gene_name'])}
        self.gene_region_ds = GenomeIntervalDataset(
            bed_file = bed_file,
            fasta_file = fasta_file,
            filter_df_fn = lambda x: x,
            return_seq_indices = False,
            shift_augs = (0,0),
            rc_aug = False,
            return_augs = True,
            context_length = context_length,
            chr_bed_to_fasta_map = {}
        )
        self.transcriptome = Transcriptome(gtf_file)

    def __len__(self):
        return len(self.snp_df) * 2

    def __getitem__(self,idx):
        # get variant
        allele = 'Ref' if idx < len(self.snp_df) else 'Alt'
        rec = self.snp_df.iloc[idx % len(self.snp_df)]
        gene = rec['gene_name']
        strand = rec['Strand']
        pos = rec['Pos'] - rec['Start']# - 1
        # get sequence of associated gene
        gene_idx = self.gene_to_idx[gene]
        seq = self.gene_region_ds[gene_idx][0]
        # if ref, compute offset, check nuc is correct
        nuc = str_to_one_hot(rec[allele]).squeeze()
        if allele == 'Ref':
            assert torch.allclose(seq[pos], nuc), gene
        # if alt, compute offset, insert variant
        else:
            seq[pos] = nuc
        # get bins (NB: here we only extract bins from central 196kb)
        gene_slice, gene_strand = get_gene_slice_and_strand(self.transcriptome, 
                                                            gene, 
                                                            rec['Start'] + 163840,
                                                            span=True, sliced=True)
        assert strand == gene_strand, gene
        assert gene_slice.shape[0] > 0, gene # we do not want an empty span
        # make sequence sense
        if strand == '-':
            seq = seq.flip(dims=(0,1))
            gene_slice = 6143 - gene_slice
        return seq, gene_slice

# scooby predict function
def predict(model, seqs, seqs_rev_comp, conv_weights, conv_biases, bins_to_predict = None):
    bs = seqs.shape[0]
    assert bs == 1
    with torch.no_grad():
        with torch.autocast("cuda"):
            outputs = model.forward_sequence_w_convs(seqs, conv_weights, conv_biases, bins_to_predict = bins_to_predict)
            if bins_to_predict is not None:
                outputs_rev_comp = model.forward_sequence_w_convs(seqs_rev_comp, conv_weights, conv_biases, bins_to_predict = (6143 - bins_to_predict))
            else:
                outputs_rev_comp = model.forward_sequence_w_convs(seqs_rev_comp, conv_weights, conv_biases, bins_to_predict = None)
    flipped_version = torch.flip(outputs_rev_comp,(1,-3))
    outputs_rev_comp = fix_rev_comp_multiome(flipped_version)
    return (outputs + outputs_rev_comp)/2

# Make model
device = 'cuda'
scooby = Scooby.from_pretrained(pretrained_path, 
                                cell_emb_dim=14, n_tracks=3,
                                use_transform_borzoi_emb = True,
                               )
scooby.eval()
scooby.to(device)
#print(scooby)

# Prepare cell stuff
embedding = pd.read_parquet(embedding_path)
cell_type_index = pd.read_parquet(cell_type_idx_path)
cell_type_index['size'] = cell_type_index['cellindex'].apply(lambda x: len(x))
cell_type_index['celltype_name'] = cell_type_index['celltype'].copy()
cell_type_index['celltype'] = cell_type_index['celltype'].str.replace(' ', '_').replace(r"G/M_prog", "G+M_prog").replace("MK/E_prog", "MK+E_prog")
cell_type_index = cell_type_index.sort_values('celltype')
cell_type_index = cell_type_index.reset_index(drop=True)
cell_indices  = []
size_factors_per_ct = []
for _, row in tqdm.tqdm(cell_type_index.iterrows(),disable = True):
    cell_indices.append(
        torch.from_numpy(
            np.vstack(
                embedding.iloc[row['cellindex']]['embedding'].values # gets embeddings of all cells of the cell type
                )
            ).unsqueeze(0)
        ) # prep cell_embeddings
# get conv weights and biases for all cells sorted by cell type in a list
cell_emb_conv_weights_and_biases = []
for cell_emb_idx in tqdm.tqdm(cell_indices, disable = True):
    cell_emb_idx = cell_emb_idx.to(device)
    conv_weights, conv_biases = scooby.forward_cell_embs_only(cell_emb_idx)
    cell_emb_conv_weights_and_biases.append((conv_weights.to(torch.float16), conv_biases.to(torch.float16)))

# Make dl
var_ds = VariantDataset(dataset, gtf_file, fasta_file, bed_file)
var_dl = DataLoader(var_ds, shuffle=False, batch_size=1, num_workers=1, pin_memory=True)

# Run
# Run
preds = []

with torch.inference_mode():
    with torch.autocast(device):
        for batch in tqdm.tqdm(var_dl):
            seq, bins = batch
            seq, bins = seq.to(device), bins.to(device)
            seq = seq.permute(0,2,1)
            bins = bins.squeeze()
            counts_outputs_rna = get_pseudobulk_count_pred(scooby, seq, gene_slice=bins, 
                                                     strand='+', predict=predict, clip_soft=5, 
                                                     model_type = "multiome_rna",
                                                     cell_emb_conv_weights_and_biases=cell_emb_conv_weights_and_biases).cpu()
            #print(counts_outputs_rna.shape)
            counts_outputs_rna = torch.log(counts_outputs_rna+1)
            preds.append(counts_outputs_rna)

preds = torch.stack(preds)
snp_effects = (preds[len(var_ds.snp_df):] - preds[:len(var_ds.snp_df)])

adata = ad.AnnData(snp_effects.numpy(), 
                   obs=var_ds.snp_df.copy(),
                   var=cell_type_index[['celltype_name']].set_index('celltype_name'),
                  )
adata.write(out_path, compression="gzip")
