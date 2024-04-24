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

def getStat(stat, block):
    if stat == "time":
        stats = block["step"].to_list()
    elif stat == ">0":
        stats = (block["step"] > 0).astype("int").to_list()
    elif stat == "correct":
        stats = block["correct"].astype("int").to_list()
    elif stat == "count":
        stats = [1 for xx in range(len(block))]
    else:
        assert(0)

    return stats

if __name__ == '__main__':
    args = parseArgs()
    
    data = pd.read_csv(args.stats)
    data.features = data.features.map(lambda xx: frozenset(eval(xx)))
    data = data.loc[data["policy"] == args.policy,]

    scores = defaultdict(list)
    scores2 = defaultdict(list)
    collate = "featureset"
    stat = "time"
    stat2 = "correct"
    plot = "scatter"

    if plot != "text":
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

    for key, block in data.groupby("features"):
        stats = getStat(stat, block)
        stats2 = getStat(stat2, block)

        if collate == "feature":
            for fi in key:
                scores[fi] += stats
                scores2[fi] += stats2
        elif collate == "featureset":
            scores[";".join(sorted(list(key)))] += stats
            scores2[";".join(sorted(list(key)))] += stats2
        else:
            assert(0)

    #sort both into lexicographic order
    s2lst = sorted(list(scores2.items()), key=lambda xx: xx[0])
    slst = sorted(list(scores.items()), key=lambda xx: xx[0])
    labels = [feat for (feat, score) in slst]

    if plot in ["bar", "boxplot", "text"]:
        #resort into ascending by stat2
        inds = np.argsort([np.mean(sl) for sl in s2lst])
        s2lst = [s2lst[ii] for ii in inds]
        slst = [slst[ii] for ii in inds]
        labels = [labels[ii] for ii in inds]

    #drop label
    scoreLst = [score for (feat, score) in slst]
    s2lst = [score for (feat, score) in s2lst]

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
    elif plot == "scatter":
        #print(s2lst)
        #print(scoreLst)
        axis = [np.mean(sl) for sl in s2lst]
        value = [np.mean(sl) for sl in scoreLst]
        plt.scatter(axis, value)
        for li, xx, yy in zip(labels, axis, value):
            plt.text(xx, yy, li)

        plt.title(args.policy)
        plt.xlabel(stat2)
        plt.ylabel(stat)
        plt.show()


