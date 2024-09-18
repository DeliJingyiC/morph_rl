from pathlib import Path
import pandas as pd
import json
import csv
import numpy as np
import random
from scipy import optimize
from syncretism import readUDVocab, fullCollapseSyncretism, relabelCells
from arguments import parseArgs

if __name__ == "__main__":
    args = parseArgs()

    data_path = Path(args.project + "/rl/dataset/")

    data = pd.read_csv(args.ud_dataframe)

    language = Path(args.ud_dataframe).name.replace("ud_", "").replace(".csv", "")

    lemmas = set(data.lemma)
    #print("all lemmas", lemmas)
    print("known lemmas", len(lemmas))

    nDev = 200
    nTest = 400
    nTrain = len(lemmas) - nDev - nTest

    shuffled = list(lemmas)
    np.random.shuffle(shuffled)

    train = shuffled[:nTrain]
    test = shuffled[nTrain:(nTrain + nTest)]
    dev = shuffled[(nTrain+nTest):]

    devData = data[data.lemma.isin(dev)]
    testData = data[data.lemma.isin(test)]
    trainData = data[data.lemma.isin(train)]

    print(len(trainData), "training items")
    print(len(devData), "dev items")
    print(len(testData), "test items")

    devData.to_csv(data_path / f"{language}_inflection_split_dev.csv", index=False)
    testData.to_csv(data_path / f"{language}_inflection_split_test.csv", index=False)
    trainData.to_csv(data_path / f"{language}_inflection_split_train.csv", index=False)

