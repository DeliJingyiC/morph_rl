#!/bin/bash
#SBATCH --time=55:00:00
#SBATCH --account=PAS1957
#SBATCH --output=output/%j.log
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=2

# source .pitzerenv/bin/activate
cwd=$(pwd)

python3 build-dataset.py /users/PAS2062/delijingyic/project/morph/neural-transducer/2022InflectionST/part1/development_languages ang_small.train
