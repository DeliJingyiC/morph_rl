import sys
import pandas as pd
from collections import *
from pathlib import Path
import re


class Clusters:
    def __init__(self, points):
        self.members = {}
        for pi in points:
            self.members[pi] = set([pi])
        self.pointToID = {}
        for pi in points:
            self.pointToID[pi] = pi

    def merge(self, pi, pj):
        idI = self.trace(pi)
        idJ = self.trace(pj)
        if idI == idJ:
            return
        self.pointToID[pi] = idJ
        self.members[idJ].update(self.members[idI])
        del self.members[idI]

    def trace(self, pi):
        parent = pi
        while self.pointToID[parent] != parent:
            parent = self.pointToID[parent]
        return parent

    def __str__(self):
        def strfy(feats):
            return ";".join(feats)

        return "\n".join(
            [f"{strfy(key)}\t:\t\t {',   '.join([strfy(vi) for vi in vals])}"
             for (key, vals) in
             self.members.items()])
    
def findPossSyncretism(pos2form):
    sync = defaultdict(set)
    for pi, fi in pos2form.items():
        sync[fi.lower()].add(pi)

    return sync

def collapseSyncretism(lemmaPosForm, cutoff):
    allPos = set()
    for lemma, pos2form in lemmaPosForm.items():
        for pos in pos2form:
            allPos.add(pos)

    syncsExcluded = defaultdict(Counter)
    syncsAllowed = defaultdict(Counter)
    
    for lemma, pos2form in lemmaPosForm.items():
        possSyncs = findPossSyncretism(pos2form)

        # for pi, fi in pos2form.items():
        #     print(pi, fi)

        # print()

        # for fi, syncset in possSyncs.items():
        #     if len(syncset) > 1:
        #         print(fi, syncset)

        # print()

        for pi, fi in pos2form.items():
            syncset = possSyncs[fi]
            for pj in syncset:
                syncsAllowed[pi][pj] += 1
                syncsAllowed[pj][pi] += 1

            for pj in pos2form:
                if pj not in syncset:
                    syncsExcluded[pi][pj] += 1
                    syncsExcluded[pj][pi] += 1

    cells = Clusters(allPos)
    for pi in sorted(allPos, key=len, reverse=True):
        for pj, ct in sorted(syncsAllowed[pi].items(),
                         key=lambda xx: xx[1], reverse=True):
            if pi == pj:
                continue
            if syncsExcluded[pi][pj] < cutoff:
                cells.merge(pi, pj)
                break

    return cells

def readConll(fh, multiword=False):
    eos = True
    for line in fh:
        if not line.strip() or line.startswith("#"):
            if not eos:
                yield (None, None, None, None)
                eos = True

            continue

        eos = False
        flds = line.split("\t")
        idn, word, lemma, uPos, aPos, posFeats = flds[0:6]
        if "-" in idn and not multiword:
            continue
        posFeats = [xx.split("=") for xx in posFeats.split("|") if xx != "_"]
        posFeats = dict(posFeats)
        yield (word, lemma, uPos, posFeats)

if __name__ == "__main__":
    data_path=Path(
    "/users/PAS2062/delijingyic/project/morph/rl/sq_output")
    file_path = Path(
   "/users/PAS2062/delijingyic/project/morph/rl/dataset/ud_train.csv"
)
    df = pd.read_csv(file_path,sep=",")
    lemmaPosForm = defaultdict(dict)
    # print(df)
    # exit(0)
    feat_list_1=['form','lemma','upos','feats','frequency','Log_Freq_HAL','I_Mean_RT']
    df_new=pd.DataFrame(columns=feat_list_1)
    for i, row in df['feats'].items():
        new_line=[]
        new_line.append(df.loc[i,'form'])
        new_line.append(df.loc[i,'lemma'])
        new_line.append(df.loc[i,'upos'])

        lis_row=[]
        row=row.split(',')
        for irow in row:
            # print(irow)
            it=irow.split(':')[-1]
            # print(it)
            it=''.join(filter(str.isalpha,it))
            # it=it.replace('\'','')
            if it=="Yes":
                continue
            lis_row.append(it)
            if '' in lis_row:
                lis_row.remove('')
        new_line.append(lis_row)
        new_line.append(df.loc[i,'frequency'])
        new_line.append(df.loc[i,'Log_Freq_HAL'])
        new_line.append(df.loc[i,'I_Mean_RT'])
        # input(new_line)
        df_new.loc[len(df_new.index)]=new_line

        for i_, row_ in df_new['feats'].items():
            posFeats = frozenset(row_)
            # print(df_new.loc[i_,'lemma'])
            # input(df_new.loc[i_,'form'])
            lemma=df_new.loc[i_,'lemma']
            form=df_new.loc[i_,'form'].lower()
        # print(posFeats)
            lemmaPosForm[lemma][posFeats] = form
        # print(lemmaPosForm)
        # input()
    cellCounts = Counter()
    for li, pos2form in lemmaPosForm.items():
        for pos in pos2form:
            cellCounts[pos] += 1

    #get rid of cells that don't look reasonable
    for li, pos2form in lemmaPosForm.items():
        newPos2Form = dict([ (key, val) for (key, val) in pos2form.items()
                        if cellCounts[key] >= 5])
        lemmaPosForm[li] = newPos2Form

    cells = collapseSyncretism(lemmaPosForm, cutoff=5)

    # for cell, values in cells.members.items():
    #     print(cell, "\t", values)
    #     print()
        
    cellNames = {}
    for cell, vals in cells.members.items():
        if vals is not None:
            # if these cells share some features, assign those as a name
            name = frozenset.intersection(*vals)
            if len(name) == 0:
                # otherwise assign a name arbitrarily
                name = list(vals)[0]

            for vi in vals:
                cellNames[vi] = name

    # for each syncretism set, print all matching forms and the name
    for cell, vals in cells.members.items():
        if vals is not None:
            print(vals, "belong to", cellNames[cell])
            print()

