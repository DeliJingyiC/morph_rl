import sys
import os
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

class PolStats:
    def __init__(self, name):
        self.name = name
        self.corrects = []
        self.times = []
        self.steps = []
        self.byFeats = defaultdict(list)

    def addStats(self, step, time, correct, feats, verbose=False):
        if step > 0 and verbose is True:
            print("\t", "waited till step", step, "for", inp, feats, trg)

        self.byFeats[feats].append((step, correct))
        self.corrects.append(correct)
        self.times.append(time)
        self.steps.append(step)

    def report(self):
        self.corrects = np.array(self.corrects)
        self.times = np.array(self.times)
        self.steps = np.array(self.steps)

        print("results of policy", self.name)
        print(np.sum(self.corrects), "/", len(self.corrects), np.mean(self.corrects), "items correct")
        print(np.mean(self.times), "time taken")
        print(np.sum((self.steps > 0)), "steps > 0")
        print(np.mean(self.steps), "avg step")

    def toDataframe(self):
        res = []
        for feats, outcomes in self.byFeats.items():
            for step, correct in outcomes:
                row = { "features" : set(feats), 
                        "policy" : self.name,
                        "step" : step,
                        "correct" : correct }
                res.append(row)

        return pd.DataFrame.from_records(res)

    def featureReport(self):
        def rate1(item):
            key, lst = item
            lst = [time for (time, correct) in lst]
            nPlus = len([xx for xx in lst if xx > 0])
            return nPlus / len(lst)

        print("featural analysis of", self.name)
        for ft, ts in sorted(self.byFeats.items(), key=rate1, reverse=True):
            ts = [time for (time, correct) in ts]
            if len(ts) <= 1:
                continue

            ftS = ";".join(sorted(ft))

            nPlus = len([xx for xx in ts if xx > 0])
            m1 = np.mean(ts)
            m2 = np.median(ts)
            print(f"{ftS}\t#: {len(ts)}\t#>0: {nPlus} ({nPlus / len(ts)})\tavg: {m1}\tmed: {m2}")
        print()

def describePolicies(aql, verbose=False, outfile=None):
    policies = ["predicted", "optimal", "stop", "wait"]
    polStats = {}
    for policy in policies:
        polStats[policy] = PolStats(policy)

    for ind, key in enumerate(aql.trainKeys):
        (lemma, form, feats), block = key
        rewards, _ = aql.simulator.simulate(key)
        if rewards.loc[0, "optimal_action"] == "wait":
            prs = rewards.loc[:, "pred_reward_stop"].to_numpy()
            prw = rewards.loc[:, "pred_reward_wait"].to_numpy()
            prs = np.exp(prs) / (np.exp(prs) + np.exp(prw))
            rewards["pred_reward_stop"] = prs
            rewards["pred_reward_wait"] = 1 - prs
            print(rewards.loc[:,["correct", "pred_reward_stop", "pred_reward_wait", "predicted_action", "optimal_action"]])

        for policy in policies:
            (step, time, correct) = aql.simulator.evaluatePolicy(rewards, policy)
            polStats[policy].addStats(step, time, correct, feats, verbose)

    for policy in policies:
        polStats[policy].report()
        print()

    if verbose == "features":
        polStats["predicted"].featureReport()
        print()
        print()
        polStats["optimal"].featureReport()

    if outfile:
        dfs = [stats.toDataframe() for stats in polStats.values()]
        df = pd.concat(dfs)
        df = df.reset_index()
        df.to_csv(outfile)

if __name__ == '__main__':
    args = parseArgs()
    dataDir = Path(args.project) / "rl/dataset/"
    neuralDir = Path(args.project) / "neural-transducer/data/reinf_inst"

    checkpoints = Path(args.project) / "neural-transducer/checkpoints"

    cumulative = (not args.noncumulative and not args.single_source)
    language = args.language

    checkpoint = checkpoints / language

    warnings.filterwarnings("ignore")
    torch.set_warn_always(False)

    split = "test"
    dataPath = dataDir / (f"query_{language}_{split}.csv")

    epoch = args.epoch

    aql = AdaptiveQLearner(mode="load", train=dataPath,
                           load_model=checkpoint/language, load_epoch=epoch)

    verbose = "features"

    outfile = checkpoint / (f"statistics_{epoch}_{split}.csv")

    describePolicies(aql, verbose=verbose, outfile=outfile)
