import numpy as np
from os.path import join
from pathlib import Path
import pandas as pd
import re
from random import choice
from arguments import parseArgs

def powerset(s):
    x = len(s)
    lis = []
    for i in range(1, 1 << x):
        lis.append([s[j] for j in range(x) if (i & (1 << j))])
    return lis

class DataGenerator:
    def __init__(self, data, rtData, logFreqCol="Log_Freq_HAL"):
        super(__class__, self).__init__()
        self.data = data.copy()
        self.logFreqCol = logFreqCol

        allFeats = set()
        for fset in self.data.feats:
            allFeats.update(fset)

        #print("know about these feats:", allFeats)

        for fi in allFeats:
            self.data[fi] = self.data.feats.map(lambda xx: fi in xx)

        #print(self.data[:5])

        self.regressor, self.y1_std = self.computeRegressor(rtData)
        self.low_bound_log_freq = self.data[self.logFreqCol].sum()
        self.low_bound_rt = self.regressor(self.low_bound_log_freq)

    def computeRegressor(self, elp):
        elp = elp[elp["I_Mean_RT"] != ""]
        elp["I_Mean_RT"] = elp["I_Mean_RT"].map(float)
        elp["Log_Freq_HAL"] = elp["Log_Freq_HAL"].map(float)

        y1_std = np.std(elp["I_Mean_RT"])
        x = elp["Log_Freq_HAL"]
        y = elp["I_Mean_RT"]
        z = np.polyfit(elp["Log_Freq_HAL"], elp['I_Mean_RT'], 1)

        a = np.poly1d(z, r=False, variable=["x"])
        return a, y1_std

    def findMatches(self, query, verbose=False):
        if query is None:
            return None

        if verbose:
            print("matches for", query)

        results = self.data
        for qcol, qval in query:
            if verbose:
                print("filtering so that", qcol, "==", qval)
            results = results.loc[results[qcol] == qval]

            if verbose:
                print(results[:5])

        return results

    def makeQueries(self, targetLemma, targetFeats):
        #print("make queries", targetLemma, targetFeats)
        queryable = [("lemma", targetLemma)] + [(xx, True) for xx in targetFeats]
        queries = powerset(queryable)
        queries = [None,] + queries

        entries = []

        for query in queries:
            matches = self.findMatches(query)
            #print(query, "found", len(matches), "matching")
            response = self.responseTimeFromMatches(matches)
            if matches is not None:
                source = matches.sample(n=1)
                size = len(matches)
                source = { "lemma" : source["lemma"].squeeze(),
                           "feats" : source["feats"].squeeze(),
                           "form" : source["form"].squeeze(),}
            else:
                source = {"lemma": False, "feats" : False, "form" : False}
                size = 0
                query = False

            entries.append( { "query" : query,
                              "matched_set_size" : size,
                              "response_time" : response,
                              "source_lemma" : source["lemma"],
                              "source_feats" : source["feats"],
                              "source_form" : source["form"],
                          } )

        return entries

    def responseTimeFromMatches(self, matches):
        if matches is None:
            return 0
        #print(matches)
        sumLogFreq = matches[self.logFreqCol].sum()
        mean_rt_predict = self.regressor(sumLogFreq)
        #print("sampled mean", mean_rt_predict)
        item_rt = np.random.normal(mean_rt_predict, self.y1_std)
        #print("rt", item_rt)
        item_rt -= self.low_bound_rt
        return item_rt

    def allQueries(self, data):
        makeQ = lambda row : self.makeQueries(row["lemma"], row["feats"])
        data["Q"] = data.apply(makeQ, axis=1)
        data = data.explode("Q")
        data = data.reset_index(drop=True) #explode creates duplicate indices
        QQ = data["Q"]
        q1 = QQ[0] #get list of columns
        for key in q1:
            data[key] = QQ.map(lambda dd: dd[key])
        data = data.drop(columns=["Q",])
        return data

if __name__ == '__main__':
    args = parseArgs()
    dataDir = Path(args.project) / "rl/dataset/"
    language = args.language

    dsets = []
    for split in ["train", "dev", "test"]:
        dataPath = dataDir / (f"{language}_inflection_split_{split}.csv")
        data = pd.read_csv(dataPath)
        data.feats = data.feats.map(eval)
        dsets.append(data)

    data = pd.concat(dsets, axis=0, ignore_index=True)
    #print(data)
    rtPath = Path(args.project) / "rl/dataset/elp_withsublex.csv"
    rtData = pd.read_csv(rtPath)
    if "Log_Freq_HAL" in data:
        col = "Log_Freq_HAL"
    else:
        col = "local_log_freq"
    datagenerator = DataGenerator(data, rtData, logFreqCol=col)

    for split, dset in zip(["train", "dev", "test"], dsets):
        dset = datagenerator.allQueries(dset)
        dset.to_csv(dataDir / (f"query_{language}_{split}.csv"), index=False)
