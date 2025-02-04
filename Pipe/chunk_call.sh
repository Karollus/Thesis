#!/bin/bash

CONFIG=$1
job_names=$2
output_files=$3

SNAKEFILE=/s/project/multispecies/Modelling/Snoopy/Pipe/chunk_pipe.snake
number_of_snakemake_jobs=12
number_of_snakemake_cores=640
partition=urgent

echo $SNAKEFILE
echo $CONFIG
echo $job_names
echo $output_files

/opt/modules/i12g/anaconda/envs/karollus-mamba/bin/snakemake \
           -s $SNAKEFILE \
           --keep-going \
           --default-resources ntasks=1 cpu=1 mem_mb=30000 gpu=0 "time='24:00:00'" \
           --cluster "sbatch \
                     --requeue \
                     --partition=$partition \
                     --ntasks {resources.ntasks} \
                     --cpus-per-task {resources.cpu} \
                     --parsable \
                     --mem {resources.mem_mb}M \
                     --job-name=$job_names-{rule} \
                     --error $output_files{rule}.%J.err --output $output_files{rule}.%J.out \
                     --gres=gpu:a40:{resources.gpu} \
                     --time={resources.time} \
                     " \
          --cores $number_of_snakemake_cores \
          --use-conda \
          --retries 0 \
          -j $number_of_snakemake_jobs \
          --configfile $CONFIG \
          --rerun-triggers mtime \
          --nolock \