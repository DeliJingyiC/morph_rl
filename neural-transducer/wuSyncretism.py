import sys
import pandas as pd
from pathlib import Path
from collections import *

def findPossSyncretism(pos2form):
    sync = defaultdict(set)
    for pi, fi in pos2form.items():
        sync[fi.lower()].add(pi)

    return sync

def collapseSyncretism(lemmaPosForm):
    allPos = set()
    for lemma, pos2form in lemmaPosForm.items():
        for pos in pos2form:
            allPos.add(pos)

    syncsExcluded = defaultdict(Counter)

    for lemma, pos2form in lemmaPosForm.items():
        possSyncs = findPossSyncretism(pos2form)

        # for pi, fi in pos2form.items():
        #     print(pi, fi)

        # print()

        # for fi, syncset in possSyncs.items():
        #     print(fi, syncset)

        # print()

        for pi, fi in pos2form.items():
            syncset = possSyncs[fi]
            for pj in pos2form:
                if pj not in syncset:
                    syncsExcluded[pi][pj] += 1
                    syncsExcluded[pj][pi] += 1

    cells = defaultdict(list)
    for pi in sorted(allPos, key=len):
        if pi not in cells:
            # print("finding all syncs for", pi)
            for pj in sorted(allPos, key=len):
                if syncsExcluded[pi][pj] < 1 and pj not in cells:
                    # print("adding sync between", pi, pj)
                    cells[pi].append(pj)
                    if pj != pi:
                        cells[pj] = None

    return cells


if __name__ == "__main__":
    file_path = Path(
    "/users/PAS2062/delijingyic/project/morph/rl/sq_output/query_df_after.csv"
)
    df = pd.read_csv(file_path,sep=",")
    df["qLen"] = df["QUERY"].map(len)

    lemmaPosForm = defaultdict(dict)

    ### construct a dictionary mapping { lemma : { set(features) : form } }
    for inp, block in df.groupby("INPUT"):
        inp = eval(inp)
        if "_" in inp:
            continue #why the heck is _ treated a word?
        idx = block["qLen"].idxmax()
        target = block.loc[[idx,], ["RESPONSE_FORM",]].values[0][0]
        lemma = inp[0]
        feats = inp[1:]
        #English: remove "being" from set
        if lemma != "be":
            lemmaPosForm[lemma][frozenset(feats)] = target

    # find set of forms which are predictably identical across all lemmas
    cells = collapseSyncretism(lemmaPosForm)
    cellNames = {}
    for cell, vals in cells.items():
        if vals is not None:
            # if these cells share some features, assign those as a name
            name = frozenset.intersection(*vals)
            if len(name) == 0:
                # otherwise assign a name arbitrarily
                name = list(vals)[0]

            for vi in vals:
                cellNames[vi] = name

    # for each syncretism set, print all matching forms and the name
    for cell, vals in cells.items():
        if vals is not None:
            print(vals, "belong to", cellNames[cell])
            print()
