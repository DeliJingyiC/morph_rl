#!/bin/bash
module load python/3.9-2022.05
rm -rf .workenv
python -m venv .workenv --upgrade-deps

source .workenv/bin/activate
python -m pip install pandas yapf pyconll conllu
