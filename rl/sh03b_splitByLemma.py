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

    lemmas = set(data.lemma)
    #print("all lemmas", lemmas)
    print("known lemmas", len(lemmas))

    nDev = 800
    nTest = 200
    nTrain = len(lemmas) - nDev - nTest

    shuffled = list(lemmas)
    np.random.shuffle(shuffled)

    dev = shuffled[:nDev]
    test = shuffled[nDev:(nDev + nTest)]
    train = shuffled[(nDev+nTest):]

    devData = data[data.lemma.isin(dev)]
    testData = data[data.lemma.isin(test)]
    trainData = data[data.lemma.isin(train)]

    devData.to_csv(data_path / "ud_inflection_split_dev.csv", index=False)
    testData.to_csv(data_path / "ud_inflection_split_test.csv", index=False)
    trainData.to_csv(data_path / "ud_inflection_split_train.csv", index=False)

