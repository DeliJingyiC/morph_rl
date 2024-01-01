import sys
import pandas as pd
from collections import *
from pathlib import Path

def processFeats(feats, lemma, featsep):
    # return featsep.join([xx for xx in feats])
    return featsep.join([xx for xx in feats if xx != lemma])


def writeRow(inp, sources, target,
             featsep=";", lsep=".", infsep=">", instsep=":"):
    # for l, q, f in sources:
        # query =q
        # print("query",q)
        # input()
    feats = inp[1:]
    lemma = inp[0]
    # if lemma == 'get':
    #     print('lemma',lemma)
    #     print('feats',feats)
    #     input()

    # sourceStrings = [
    #     f"{srcLemma}{lsep}{processFeats(srcFeats, srcLemma, featsep)}{infsep}{srcForm}" for
    #     (srcLemma, srcFeats, srcForm)
    #     in sources]
    sourceStrings = [
        f"{srcLemma}{lsep}{processFeats(srcFeats, srcLemma, featsep)}{infsep}{srcForm}" for
        (srcLemma, srcFeats, srcForm)
        in sources]
    fullStr = f"{lemma}{lsep}{processFeats(feats, lemma, featsep)}{instsep}{instsep.join(sourceStrings)}"
    return fullStr

if __name__ == "__main__":
    data_path=Path(
    "/users/PAS2062/delijingyic/project/morph/rl/sq_output")
    file_path = Path(
    "/users/PAS2062/delijingyic/project/morph/rl/sq_output/query_df_train.csv"
)
    df = pd.read_csv(file_path,sep=",")
    
    df["qLen"] = df["QUERY"].map(len)
    # print(df["qLen"])
    # exit(0)         
    for inp, block in df.groupby("INPUT"):
        inp = eval(inp)
        if "_" in inp:
            continue #why the heck is _ treated a word?

        idx = block["qLen"].idxmax()
        target = block.loc[[idx,], ["RESPONSE_FORM",]].values[0][0]
        timet = block.loc[[idx,], ["TIME",]].values[0][0]

        # if target == 'contributed':
        #         print(block["qLen"].idxmax())
        #         print(block.loc[[idx,], ["RESPONSE_FORM",]].values[0])
            # print(block.loc[[idx,], ["RESPONSE_FORM",]].values[0][0])
            # print(block.loc[[idx,], ["RESPONSE_LEMMA",]].values[0][0])
            # print(block.loc[[idx,], ["INPUT",]].values[0][0])



        #         input()
            # print("inp",inp)
            # print("idx",idx)

        # print("block",block)
            # input()
        #this line prints the Wu-like instance with no multisource
        inst = writeRow(inp, [], target)
        # if target == 'contributed':
            # print()
        print(",".join([inst, target, "0"]))
        sources = []

        #this block prints the remaining instances with multisource
        for ind, row in block.sort_values("TIME").iterrows():
            # print("eval(row)",eval(row["QUERY"]))
            # input()
            sources.append([
                            row["RESPONSE_LEMMA"],
                            eval(row["QUERY"]),
                            row["RESPONSE_FORM"]])
            # print('sources',sources)
            inst = writeRow(inp, sources, target)
            # print('inst',inst)
            print(",".join([inst, target, "0",str(timet)]))
            # input()

