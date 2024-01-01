#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --account=PAS1957
#SBATCH --output=output/%j.log
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=2

cwd=$(pwd)

cd $TMPDIR
module load python/3.9-2022.05

source /users/PAS2062/delijingyic/project/morph/.pitzermorphenv/bin/activate
cd $cwd
python3 sh02_wuFromUDSyncretism.py
