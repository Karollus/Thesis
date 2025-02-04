import os
import glob
import yaml
import argparse

import numpy as np
import pandas as pd
import polars as pl
import pyranges as pr
import pyfaidx
import scipy
import torch
import tqdm
import anndata as ad
from accelerate import Accelerator
import pyranges
import plotnine as p9

from torch.utils.data import DataLoader, Dataset

from enformer_pytorch.data import GenomeIntervalDataset, str_to_one_hot

#from scooby.utils.transcriptome import Transcriptome
#from scooby.utils.utils import get_gene_slice_and_strand
from borzoi_pytorch import AnnotatedBorzoi, Transcriptome
from borzoi_pytorch.config_borzoi import BorzoiConfig

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
out_dir = "/"+os.path.join(*out_path.split("/")[:-1])
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
idx = out_path.split(".")[-2].split("_")[-2:]
start_idx = int(idx[0])
end_idx = int(idx[1])
print(start_idx, end_idx)
table = pd.read_table(in_path)
bed_cols = ['chrom','startTSS','endTSS','measuredGeneSymbol','chromMid','name']
table = table.iloc[start_idx:end_idx+1]
table = table[bed_cols]
bed_file = os.path.join(out_dir,f'bed_{start_idx}_{end_idx}.tsv')
table.to_csv(bed_file,sep="\t",index=None,header=None)
print(bed_file)

# load config
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# get config params
replicate = config['replicate']
#targets_path = config['targets_path']
gtf_file = config['gtf_file']
fasta_file = config['fasta_file']
pretrained_path = config['pretrained_path']
return_center_bins_only = config['return_center_bins_only']
disable_autocast = config.get('disable_autocast', False)

print(pretrained_path)
print(config_path)

cfg = BorzoiConfig.from_pretrained(pretrained_path)
if return_center_bins_only:
    bins = 6144
    offset = 163840
else:
    bins = 16384 - 32
    offset = 512
    cfg.return_center_bins_only = False

def get_gene_slice_and_strand(transcriptome, gene, position, span, bins=bins):
    """
    Retrieves the gene slice and strand information from the transcriptome.

    Args:
        transcriptome: The transcriptome object.
        gene (str): The name of the gene.
        position (int): The genomic position.
        span (int): The span of the genomic region.
        sliced (bool, optional): Whether to slice the output. Defaults to True.

    Returns:
        Tuple[torch.Tensor, str]: The gene slice and strand.
    """
    gene_slice = transcriptome.genes[gene].output_slice(
        position, bins * 32, 32, span=span, 
    )  # select right columns
    strand = transcriptome.genes[gene].strand
    return gene_slice, strand

class CrisprDataset(Dataset):

    def __init__(self, bed_file, gtf_file, fasta_file, 
                context_length = 524288, bins = bins):
        self.gene_region_ds = GenomeIntervalDataset(
                                bed_file = bed_file,
                                fasta_file = fasta_file,
                                filter_df_fn = lambda x: x,
                                return_seq_indices = False,
                                shift_augs = (0,0),
                                rc_aug = False,
                                return_augs = True,
                                context_length = context_length,
                                chr_bed_to_fasta_map = {},
                                #has_header=True,
                            )
        # column order ['chrom','startTSS','endTSS','measuredGeneSymbol','chromMid','name']
        self.seq_start = self.gene_region_ds.df['column_2'] - (context_length//2) + 1 # Interval of length 1 gets expanded
        self.enh_mid = self.gene_region_ds.df['column_5'] - self.seq_start
        self.transcriptome = Transcriptome(gtf_file)
        self.context_length = context_length
        self.bins = bins

    def __len__(self):
        return len(self.gene_region_ds.df)

    def __getitem__(self,idx):
        rec = self.gene_region_ds.df[idx]
        gene = rec['column_4'].item()
        pred_start = rec['column_2'].item() - ((self.bins*32)//2) + 1
        # get sequence
        seq = self.gene_region_ds[idx][0]
        # get gene slice and strand
        gene_slice, gene_strand = get_gene_slice_and_strand(self.transcriptome, 
                                                            gene, 
                                                            pred_start,
                                                            span=False)
        assert gene_slice.shape[0] > 0, gene # we do not want an empty span
        enh_mid = self.enh_mid[idx]
        assert enh_mid - 1000 >= 0
        assert enh_mid + 1000 < self.context_length
        if gene_strand == '-':
            seq = seq.flip(dims=(0,1))
            gene_slice = (self.bins - 1) - gene_slice
            enh_mid = self.context_length - 1 - enh_mid
        return seq, gene_slice, enh_mid

# Make model
device = 'cuda'
borzoi = AnnotatedBorzoi.from_pretrained(pretrained_path, config=cfg)
# subset head
k562_tracks = borzoi.tracks_df.loc[borzoi.tracks_df.description  == 'RNA:K562'].index
borzoi.set_track_subset(torch.tensor(k562_tracks))
print(borzoi)
borzoi.eval()
borzoi.to(device)

crispr_ds = CrisprDataset(bed_file, gtf_file, fasta_file)
crispr_dl = DataLoader(crispr_ds,shuffle=False, batch_size=1, num_workers=1, pin_memory=True)

# Run
preds = []

for batch in tqdm.tqdm(crispr_dl,miniters = 10):
    seq, bins, enh_mid = batch
    seq = seq.to(device)
    seq = seq.permute(0,2,1)
    seq.requires_grad = True
    with torch.autocast(device, enabled=not disable_autocast):
        # predict gene coverage
        pred = borzoi.predict_gene_count(seq, [bins.squeeze()])
        # average over relevant tracks
        pred = pred.mean(axis=-1) 
        #pred = torch.log(pred + 1)
        # compute gradient
    pred.backward()
    grads = seq.grad.clone()
    # remove mean across nucleotides (cc Peter Koo)
    grads = grads - grads.mean(axis=1, keepdims=True)
    # compute absolute reference gradient
    grads = torch.abs(grads*seq).amax(axis=1).squeeze()
    # average over region and normalize
    enh_grad = grads[enh_mid - 1000:enh_mid + 1000].mean()
    enh_grad = enh_grad - grads.mean()
    # return 
    preds.append(enh_grad.detach().cpu())

preds = torch.stack(preds)

obs_frame = crispr_ds.gene_region_ds.df.to_pandas()
obs_frame.columns = bed_cols

adata = ad.AnnData(preds.unsqueeze(-1).numpy(force=True), 
                   obs=obs_frame,
                   var=pd.DataFrame([{'var':'pred'}]),
                  )
adata.write(out_path, compression="gzip")

os.remove(bed_file)
