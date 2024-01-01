#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --account=PAS1957
#SBATCH --output=output/%j.log
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=2

source .workenv/bin/activate
cwd=$(pwd)

python3 sh02_getUD.py
