#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=PAS1957
#SBATCH --output=output/seq_ds_short.log
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=2

module load python/3.9-2022.05
source activate tfgpu

#python -u rl/sh10b_sequential_dataset.py --project `pwd` --language UD_Turkish-Kenet --split test
python -u rl/sh10c_sequential_unimorph.py --project `pwd` --language UD_Turkish-Kenet --split dev
