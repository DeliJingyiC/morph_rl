#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --account=PAS1957
#SBATCH --output=output/%j.log
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=2

cwd=$(pwd)

cd $TMPDIR
module load python/3.9-2022.05
rm -rf .pitzerenv
python -m venv .pitzerenv --upgrade-deps

source .pitzerenv/bin/activate
python3 -m pip install sqlalchemy numpy pandas
cd $cwd
python3 sh03_create_sql_tbl.py
