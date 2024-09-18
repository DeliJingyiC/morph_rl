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
        self.stores = []
        self.byFeats = defaultdict(list)

    def addStats(self, stats, feats, verbose=False):
        if stats["steps"] > 0 and verbose is True:
            print("\t", "waited till step", step, "for", inp, feats, trg)

        self.byFeats[feats].append(stats)
        self.corrects.append(stats["correct"])
        self.times.append(stats["time"])
        self.steps.append(stats["steps"])
        self.stores.append(stats["stored"])

    def report(self, ofh=sys.stdout):
        self.corrects = np.array(self.corrects)
        self.times = np.array(self.times)
        self.steps = np.array(self.steps)
        self.stores = np.array(self.stores, dtype="int")

        print("results of policy", self.name, file=ofh)
        print(np.sum(self.corrects), "/", len(self.corrects), np.mean(self.corrects), "items correct", file=ofh)
        print(np.mean(self.times), "time taken", file=ofh)
        print(np.sum((self.steps > 0)), "steps > 0", file=ofh)
        print(np.sum((self.stores > 0)), "stored anything", file=ofh)
        print(np.mean(self.steps), "avg step", file=ofh)

    def toDataframe(self):
        res = []
        for feats, outcomes in self.byFeats.items():
            for stats in outcomes:
                sources = stats["sources"]
                source1 = sources[0]
                if not isinstance(source1, list):
                    source1 = source1.to_list()
                source2 = None
                if len(sources) > 1:
                    source2 = sources[1]
                    if not isinstance(source2, list):
                        source2 = source2.to_list()

                row = { "features" : set(feats), 
                        "policy" : self.name,
                        "source1" : source1,
                        "source2" : source2,
                }

                for ki, vi in stats.items():
                    row[ki] = vi

                res.append(row)

        return pd.DataFrame.from_records(res)

    def featureReport(self, ofh=None):
        def rate1(item):
            key, stts = item
            lst = [si["steps"] for si in stts]
            nPlus = len([xx for xx in lst if xx > 0])
            return nPlus / len(lst)

        print("featural analysis of", self.name, file=ofh)
        for ft, stats in sorted(self.byFeats.items(), key=rate1, reverse=True):
            ts = [si["time"] for si in stats]
            if len(ts) <= 1:
                continue

            ftS = ";".join(sorted(ft))

            nPlus = len([si["steps"] for si in stats if si["steps"] > 0])
            m1 = np.mean(ts)
            m2 = np.median(ts)
            store = len([si["stored"] for si in stats if si["stored"]])
            print(f"{ftS}\t#: {len(ts)}\t#>0: {nPlus} ({nPlus / len(ts)})\tstore: {store}\tavg: {m1}\tmed: {m2}", file=ofh)
        print(file=ofh)

def describePolicies(aql, verbose=False, outfile=None, reportfile=None, stopAt=None):
    policies = ["predicted", "optimal", "stop", "wait"]
    polStats = {}
    for policy in policies:
        polStats[policy] = PolStats(policy)

    for ind, key in enumerate(sorted(aql.trainKeys)):
        if ind % 1 == 0:
            print(f"{ind} / {len(aql.trainKeys)}...")
        if stopAt is not None and ind > stopAt:
            break

        (lemma, form, feats), block = key
        rewards, _, _ = aql.simulator.simulate(key, train=False)

        # I don't understand why I wrote this block
        # It converts reward logits to probabilities, which should not affect policy rollout at all
        # if rewards.loc[0, "optimal_action"] == "wait":
        #     prs = rewards.loc[:, "pred_reward_stop"].to_numpy()
        #     prw = rewards.loc[:, "pred_reward_wait"].to_numpy()
        #     prs = np.exp(prs) / (np.exp(prs) + np.exp(prw))
        #     rewards["pred_reward_stop"] = prs
        #     rewards["pred_reward_wait"] = 1 - prs
        #     #print(rewards.loc[:,["correct", "pred_reward_stop", "pred_reward_wait", "predicted_action", "optimal_action"]])

        for policy in policies:
            stats = aql.simulator.evaluatePolicy(policy)
            polStats[policy].addStats(stats, feats, verbose)
            # strFeats = ";".join(sorted(list(feats)))
            # print(f"Lemma: {lemma} target form: {form} feats: {strFeats}")
            # aql.simulator.printBlock(key)
            # print(stats)

    for policy in policies:
        polStats[policy].report()
        print()

    if verbose == "features":
        polStats["predicted"].featureReport()
        print()
        print()
        polStats["optimal"].featureReport()

    if reportfile != None:
        for policy in policies:
            polStats[policy].report(ofh=reportfile)
            print(file=reportfile)

        polStats["predicted"].featureReport(ofh=reportfile)
        print(file=reportfile)
        print(file=reportfile)
        polStats["optimal"].featureReport(ofh=reportfile)

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
    if args.run == None:
        runName = language
    else:
        runName = f"{language}-{args.run}"

    checkpoint = checkpoints / runName

    warnings.filterwarnings("ignore")
    torch.set_warn_always(False)

    split = args.split
    assert(split is not None or args.force_test is not None)
    dataPath = dataDir / (f"query_{language}_{split}.csv")
    if args.force_test != None:
        dataPath = dataDir / args.force_test

    epoch = args.epoch

    aql = AdaptiveQLearner(mode="load", train=dataPath,
                           load_model=checkpoint/language, load_epoch=epoch)
    aql.model.batch_size = 256 #harmless but annoying

    if args.force_test != None:
        pass
        # cacheName = args.force_test.replace(".csv", "")
        # aql.dataHandler.readCache(Path(args.project) / f"neural-transducer/aligner_cache/{language}/{cacheName}")
    else:
        pass
        #aql.dataHandler.readCache(Path(args.project) / f"neural-transducer/aligner_cache/{language}/{split}")

    verbose = "features"

    #make report files show the correct name if you specified a special file
    if args.force_test != None:
        split = args.force_test
    outfile = checkpoint / (f"statistics_{epoch}_{split}.csv")
    report = open(checkpoint / (f"report_{epoch}_{split}.txt"), "w")

    print("Writing outputs to:", outfile, checkpoint / (f"report_{epoch}_{split}.txt"))
    describePolicies(aql, verbose=verbose, outfile=outfile, reportfile=report, stopAt=1000)
