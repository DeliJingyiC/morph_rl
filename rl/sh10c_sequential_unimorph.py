import numpy as np
from os.path import join
from pathlib import Path
import pandas as pd
import re
from random import choice
from arguments import parseArgs

from sh10b_sequential_dataset import *

if __name__ == '__main__':
    args = parseArgs()
    dataDir = Path(args.project) / "rl/dataset/"
    language = args.language

    dsets = []
    dataPath = dataDir / (f"{language}_inflection_split_train.csv")
    data = pd.read_csv(dataPath)
    data.feats = data.feats.map(eval)
    dsets.append(data)

    for split in ["dev", "test"]:
        dataPath = dataDir / (f"unimorph_{language}_short_{split}.csv")
        data = pd.read_csv(dataPath)
        data.feats = data.feats.map(eval)
        dsets.append(data)

    data = dsets[0] #train
    allData = pd.concat(dsets, axis=0, ignore_index=True)

    rtPath = Path(args.project) / "rl/dataset/elp_withsublex.csv"
    rtData = pd.read_csv(rtPath)
    if "Log_Freq_HAL" in data:
        col = "Log_Freq_HAL"
    else:
        col = "local_log_freq"
    datagenerator = DataGenerator(allData, rtData, freqCol=col)

    print("computed fits")

    for split, dset in zip(["dev", "test"], dsets[1:]):
        if split == args.split:
            print("processing", split)
            dset = datagenerator.allQueries(dset)
            dset.to_csv(dataDir / (f"query_unimorph_{language}_{split}.csv"), index=False)
