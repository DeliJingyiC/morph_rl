#!/bin/bash
#SBATCH --time=55:00:00
#SBATCH --account=PAS1957
#SBATCH --output=output/%j.log
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=2

source .pitzerenv/bin/activate
cwd=$(pwd)

python sh06_delete_yes.py
