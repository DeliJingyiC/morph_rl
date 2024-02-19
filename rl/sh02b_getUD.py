from pathlib import Path
import pandas as pd
from collections import *
import json
import csv
import numpy as np
import random
from scipy import optimize
from syncretism import readConll, readUDVocab, fullCollapseSyncretism, relabelCells, relabelCounts
from arguments import parseArgs

def dropLemma(lemmaPosForm, lemmaCell):
    for lemma, pos2form in lemmaPosForm.items():
        if lemmaCell in pos2form:
            del pos2form[lemmaCell]

def toDataframe(lemmaPosForm, localCounts):
    data = []
    for lemma, pos2form in lemmaPosForm.items():
        for pos, form in pos2form.items():
            entry = {"lemma" : lemma, "feats" : str(set(pos)), "form": form, "local_count" : localCounts[lemma][pos][form]}
            data.append(entry)
    return pd.DataFrame(data)

def mergeAndPredict(lemmaPosForm, elp_withsublex, localCounts, wordCount, useLocalFreq=False):
    elp_withsublex = elp_withsublex[elp_withsublex["I_Mean_RT"]!=""]
    elp_withsublex["I_Mean_RT"] = elp_withsublex["I_Mean_RT"].apply(lambda x: float(x) if type(x)==str else x)
    elp_withsublex["Log_Freq_HAL"] = elp_withsublex["Log_Freq_HAL"].apply(lambda x: float(x) if type(x)==str else x)

    z = np.polyfit(elp_withsublex["Log_Freq_HAL"], elp_withsublex['I_Mean_RT'],1)

    a= np.poly1d(z,r=False,variable=["x"])

    data = toDataframe(lemmaPosForm, localCounts)

    if not useLocalFreq:
        print("----initial words known:", len(data))
        merged = data.merge(elp_withsublex, left_on="form", right_on="Word", how="inner")
        print("----after merge:", len(merged))
        merged = merged.drop(columns=["Word",])
        predict_y_ud=a(merged["Log_Freq_HAL"])
        merged['I_Mean_RT'] = predict_y_ud
        return merged
    else:
        print("----initial words known:", len(data))
        #scale counts to approximate size of HAL corpus (130 mil words according to Brysbaert/New 2009)
        data["transformed_local_count"] = (data["local_count"] / wordCount) * 130e6
        data["local_log_freq"] = np.log10(1 + data["transformed_local_count"])
        predict_y_ud=a(data["local_log_freq"])
        data['I_Mean_RT'] = predict_y_ud
        return data

if __name__ == "__main__":
    args = parseArgs()

    data_path = Path(args.project + "/rl/dataset")
    data_path_ud = [Path(xx) for xx in args.ud_train]

    elp_withsublex=pd.read_csv(
        data_path /"elp_withsublex.csv",
        dtype=str,
    )

    trainPath = data_path_ud
    fullVocab = {}
    wordCount = 0
    fullCounts = defaultdict(lambda: defaultdict(Counter))
    for ti in trainPath:
        with open(ti) as fh:
            for (word, lemma, uPos, posFeats) in readConll(fh):
                if uPos != None:
                    wordCount += 1

        print("Reading", ti)
        udTrainVocab, udTrainCounts = readUDVocab(ti, args.pos_target, addLemma=True, returnCounts=True)
        for lemma, pos2form in udTrainVocab.items():
            fullVocab[lemma] = pos2form
        for lemma, pos2form in udTrainCounts.items():
            for pos, form2count in pos2form.items():
                for form, count in form2count.items():
                    fullCounts[lemma][pos][form] += count
    
    cellToCanon = fullCollapseSyncretism(fullVocab, cutoff=0)
    lemmaCell = cellToCanon[frozenset(["lemma"])]

    relabelCells(fullVocab, cellToCanon)
    relabelCounts(fullCounts, cellToCanon)
    dropLemma(fullVocab, lemmaCell)
    data = mergeAndPredict(fullVocab, elp_withsublex, fullCounts, wordCount, useLocalFreq=args.local_frequency)
    runName = "_".join([xx.parent.name for xx in trainPath])
    print("Writing results to", data_path / ("ud_%s.csv" % runName))
    data.to_csv(data_path / ("ud_%s.csv" % runName), index=False)
