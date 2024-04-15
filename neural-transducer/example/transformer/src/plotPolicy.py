import sys
import os
import math
import pandas as pd
from collections import *
from pathlib import Path
import warnings
import pickle

import numpy as np

import argparse

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats")
    parser.add_argument("--policy")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    
    data = pd.read_csv(args.stats)
    data.features = data.features.map(lambda xx: frozenset(eval(xx)))
    data = data.loc[data["policy"] == args.policy,]

    scores = defaultdict(list)
    collate = "feature"
    stat = "correct"
    plot = "text"

    if plot != "text":
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

    for key, block in data.groupby("features"):
        if stat == "time":
            stats = block["step"].to_list()
        elif stat == ">0":
            stats = (block["step"] > 0).astype("int").to_list()
        elif stat == "correct":
            stats = block["correct"].astype("int").to_list()
        else:
            assert(0)

        if collate == "feature":
            for fi in key:
                scores[fi] += stats
        elif collate == "featureset":
            scores[";".join(sorted(list(key)))] += stats
        else:
            assert(0)

    slst = sorted(list(scores.items()), key=lambda xx: len(xx[1]))
    scoreLst = [score for (feat, score) in slst]
    labels = [feat for (feat, score) in slst]

    if plot == "bar":
        plt.bar(1 + np.arange(len(labels)), height=[np.mean(sl) for sl in scoreLst])
        plt.xticks(ticks=(1 + np.arange(len(labels))), labels=labels, rotation=-45)
        plt.show()
    elif plot == "boxplot":
        plt.boxplot(scoreLst)
        plt.xticks(ticks=(1 + np.arange(len(labels))), labels=labels, rotation=-45)
        plt.show()
    elif plot == "text":
        for li, si in zip(labels, scoreLst):
            print(li, np.mean(si))



