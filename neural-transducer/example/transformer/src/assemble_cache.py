import sys
import os
import re
import functools
import math
import pandas as pd
from collections import *
from pathlib import Path
import warnings
import pickle

import numpy as np

import torch

from arguments import *

from transformer import *

from adaptive_q import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args = parseArgs()
    dataDir = Path(args.project) / "rl/dataset/"
    neuralDir = Path(args.project) / "neural-transducer/data/reinf_inst"
    checkpoints = Path(args.project) / "neural-transducer/checkpoints"

    language = args.language

    checkpoint = checkpoints / language

    warnings.filterwarnings("ignore")
    torch.set_warn_always(False)

    split = "test"
    dataPath = dataDir / (f"query_{language}_{split}.csv")

    epoch = args.epoch

    cache = Path(args.project) / f"neural-transducer/aligner_cache/{split}"

    settings = {
        "src_nb_layers" : 4,
        "src_hid_size" : 1024,
        "trg_nb_layers" : 4,
        "trg_hid_size" : 1024,
        "embed_dim" : 256,
        "nb_heads" : 4,
        "dropout_p" : .3,
        "label_smooth" : .1,
        "tie_trg_embed" : False,
        "value_mode" : "classify",
        "batch_size" : 128,
        "inference_batch_size" : 768,
        "n_sources" : 2,
        "buffer_size" : 1024, #8192,
        "n_explore" : 3,
        "epochs_per_buffer" : 1,
        "tie_value_predictor" : False,
        "value_predictor" : "conv",
        "harmony" : True,
        "aligner_cache" : None,
    }

    aql = AdaptiveQLearner(mode="create", train=dataPath, load_epoch=epoch, settings=settings)

    section = args.cache_section
    nSections = 100

    cacheTotal = len(aql.trainKeys)
    sectionLength = int(np.ceil(cacheTotal / nSections))
    sectionBegin = section * sectionLength
    sectionEnd = sectionBegin + sectionLength

    print(f"Cache section {section} from {sectionBegin} to {sectionEnd} of {cacheTotal}")

    #we need this in consistent order
    trainKeys = list(aql.train.groupby(["lemma", "form", "feats"]))
    for block in trainKeys[sectionBegin : sectionEnd]:
        aql.simulator.simulate(block)

    cacheFile = cache / f"{section}.dump"
    with open(cacheFile, "wb") as ofh:
        aql.dataHandler.writeCache(ofh)
