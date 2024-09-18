from pathlib import Path
from collections import *
import pandas as pd
import json
import csv
import numpy as np
import random
from scipy import optimize
from syncretism import readUDVocab, fullCollapseSyncretism, relabelCells
from arguments import parseArgs

def readUnimorph(unimorph):
    data = defaultdict(dict)
    with open(unimorph) as ifile:
        for line in ifile:
            (lemma, form, feats) = line.strip().split("\t")
            data[lemma][feats] = form

    return data

def getFreq(lemma, form, data):
    find = data.loc[(data.lemma == lemma) & (data.form == form),]
    if len(find) > 0:
        return find.iloc[0]
    else:
        worst = data.loc[data.local_count == 1,]
        return worst.iloc[0]

def udify(data, knownFeatures, lemmas, unimorph, pos_target):
    rows = []
    for lemma in lemmas:
        for feats, form in unimorph[lemma].items():
            if pos_target == "NOUN" and "N" not in feats.split(";"):
                continue
            if pos_target == "VERB" and "V" not in feats.split(";"):
                continue
            uFeats = set(feats.split(";"))
            if uFeats.difference(knownFeatures):
                continue

            freqInfo = getFreq(lemma, form, data)
            row = {
                "lemma" : lemma,
                "feats" : uFeats,
                "form" : form,
                "local_count" : freqInfo["local_count"],
                "transformed_local_count" : freqInfo["transformed_local_count"],
                "local_log_freq" : freqInfo["local_log_freq"],
                "I_Mean_RT" : freqInfo["I_Mean_RT"],
            }
            rows.append(row)

    return pd.DataFrame.from_records(rows)

if __name__ == "__main__":
    args = parseArgs()

    data_path = Path(args.project + "/rl/dataset/")

    data = pd.read_csv(args.ud_dataframe)
    train = pd.read_csv(args.ud_train[0])
    language = Path(args.ud_dataframe).name.replace("ud_", "").replace(".csv", "")

    unimorph = Path(args.unimorph_data)

    lemmas = set(data.lemma)
    print("known lemmas", len(lemmas))
    trainLemmas = set(train.lemma)
    lemmas = lemmas.difference(trainLemmas)
    print("known non-training lemmas", len(lemmas))

    nDev = 40
    nTest = 40

    unidat = readUnimorph(unimorph)
    ulemmas = set(unidat.keys())
    inBoth = list(ulemmas.intersection(lemmas))
    print("intersecting", len(inBoth))

    targets = []
    if args.pos_target == "VERB":
        tFeat = "V"
    elif args.pos_target == "NOUN":
        tFeat = "N"

    for lemma, sub in unidat.items():
        if any([tFeat in feats.split(";") for feats in sub]) and lemma not in trainLemmas and " " not in lemma and lemma.islower():
            targets.append(lemma)

    print("unimorph non-train lemmas with correct pos", len(targets))

    #np.random.shuffle(targets)
    targets.sort(key=lambda xx : len(xx))
    targets = targets[:2 * (nDev + nTest)]
    np.random.shuffle(targets)

    devLemmas = targets[0:nDev]
    testLemmas = targets[nDev:nDev + nTest]

    # #verbs don't have any overlap, so we'll have to take another path
    # if args.pos_target == "VERB":
    #     verbs = []
    #     for lemma, sub in unidat.items():
    #         if any(["V" in feats.split(";") for feats in sub]):
    #             verbs.append(lemma)

    #     np.random.shuffle(verbs)
    #     devLemmas = verbs[0:nDev]
    #     testLemmas = verbs[nDev:nDev + nTest]

    # else:
    #     np.random.shuffle(inBoth)

    #     devLemmas = inBoth[0:nDev]
    #     testLemmas = inBoth[nDev:nDev + nTest]

    # print(devLemmas)

    knownFeatures = set()
    for fi in data["feats"]:
        fi = eval(fi)
        knownFeatures.update(fi)
    print("known features", knownFeatures)

    devData = udify(data, knownFeatures, devLemmas, unidat, args.pos_target)
    testData = udify(data, knownFeatures, testLemmas, unidat, args.pos_target)
    devData.to_csv(data_path / f"unimorph_{language}_short_dev.csv")
    testData.to_csv(data_path / f"unimorph_{language}_short_test.csv")

