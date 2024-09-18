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
#from matplotlib import pyplot as plt
import scipy.optimize
#import piecewise_regression

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

def predictFn(hals, rts):
    def sigfn(tt, aa, bb, cc, dd):
        return aa * (1 / (1 + np.exp(bb * (tt - cc)))) + dd
    sigf, scov = scipy.optimize.curve_fit(sigfn, hals, rts, p0=[250, 1, 8, 600], maxfev=8000)
    return sigf, lambda xx: sigfn(xx, *sigf)

def mergeAndPredict(lemmaPosForm, elp_withsublex, localCounts, wordCount, n1, useLocalFreq=False):
    elp_withsublex = elp_withsublex[elp_withsublex["I_Mean_RT"]!=""]
    elp_withsublex["I_Mean_RT"] = elp_withsublex["I_Mean_RT"].apply(lambda x: float(x) if type(x)==str else x)
    elp_withsublex["Log_Freq_HAL"] = elp_withsublex["Log_Freq_HAL"].apply(lambda x: float(x) if type(x)==str else x)
    elp_withsublex["Lg10WF"] = elp_withsublex["Lg10WF"].apply(lambda x: float(x) if type(x)==str else x)

    subtlexF = elp_withsublex["Lg10WF"]
    hals = elp_withsublex["Log_Freq_HAL"]
    rts = elp_withsublex["I_Mean_RT"]

    z, predfn = predictFn(subtlexF, rts)

    print("fit returns:", z)
    tt = np.arange(np.min(subtlexF), 1.25 * np.max(subtlexF), .1)
    #a = np.poly1d(z, r=False, variable=["x"])(tt)
    a = predfn(tt)
    plt.scatter(subtlexF, rts)
    plt.plot(tt, a, "r")

    #code for comparing piecewise regression
    # ms = piecewise_regression.ModelSelection(subtlexF.tolist(), rts.tolist(),
    #                                          max_breakpoints=5, n_boot=5)

    # pwf = piecewise_regression.Fit(subtlexF.tolist(), rts.tolist(),
    #                                n_breakpoints=3, start_values=None,
    #                                verbose=True, n_boot=10)
    # pwf.summary()
    # a2 = pwf.predict(tt)
    # plt.plot(tt, a2, "c")
    #pwf.plot_fit(color="cyan")

    # plt.show()
    # assert(0)

    # a = np.poly1d(z,r=False,variable=["x"])

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
        #data["transformed_local_count"] = (data["local_count"] / wordCount) * 130e6
      
        #scale counts to size of Subtlex, 51 mil words (https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus)
        data["transformed_local_count"] = (data["local_count"] / wordCount) * 51e6
        #some simplistic Good-Turing adjustment:
        w1Rate = (n1 / wordCount)
        print("reducing counts to make room for", w1Rate, "1-count items")
        data["transformed_local_count"] *= (1 - w1Rate)

        data["local_log_freq"] = np.log10(1 + data["transformed_local_count"])
        predict_y_ud=predfn(data["local_log_freq"])
        data['I_Mean_RT'] = predict_y_ud

        # plt.scatter(data["local_log_freq"], predict_y_ud, c='c')
        # plt.show()
        # assert(0)

        return data

if __name__ == "__main__":
    args = parseArgs()

    data_path = Path(args.project + "/rl/dataset")
    data_path_ud = [Path(xx) for xx in args.ud_train]

    featStyle = args.pos_feat_style

    elp_withsublex=pd.read_csv(
        data_path /"elp_withsublex.csv",
        dtype=str,
    )

    trainPath = data_path_ud
    fullVocab = {}
    wordCount = 0
    wordFreq = Counter()
    fullCounts = defaultdict(lambda: defaultdict(Counter))
    for ti in trainPath:
        with open(ti) as fh:
            for (word, lemma, uPos, posFeats) in readConll(fh, featStyle=featStyle):
                if uPos != None:
                    wordCount += 1
                    wordFreq[word] += 1

        print("Reading", ti)
        udTrainVocab, udTrainCounts = readUDVocab(ti, args.pos_target, addLemma=True, 
                                                  returnCounts=True, featStyle=featStyle)
        for lemma, pos2form in udTrainVocab.items():
            fullVocab[lemma] = pos2form
        for lemma, pos2form in udTrainCounts.items():
            for pos, form2count in pos2form.items():
                for form, count in form2count.items():
                    fullCounts[lemma][pos][form] += count

    n1 = sum([int(freq == 1) for (word, freq) in wordFreq.most_common()])
    
    cellToCanon = fullCollapseSyncretism(fullVocab, cutoff=0)
    lemmaCell = cellToCanon[frozenset(["lemma"])]

    relabelCells(fullVocab, cellToCanon)
    relabelCounts(fullCounts, cellToCanon)
    dropLemma(fullVocab, lemmaCell)
    data = mergeAndPredict(fullVocab, elp_withsublex, fullCounts, wordCount, n1, useLocalFreq=args.local_frequency)
    runName = "_".join([xx.parent.name for xx in trainPath])
    if args.pos_target != "NOUN":
        runName += f"_{args.pos_target}"
    print("Writing results to", data_path / ("ud_%s.csv" % runName))
    data.to_csv(data_path / ("ud_%s.csv" % runName), index=False)
