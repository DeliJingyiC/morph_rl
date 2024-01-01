import sys
import pandas as pd
from pathlib import Path
import re
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
    data_path=Path(
    "/users/PAS2062/delijingyic/project/morph/rl/sq_output")
    file_path = Path(
    "/users/PAS2062/delijingyic/project/morph/rl/dataset/ud_train.csv"
)
    df = pd.read_csv(file_path,sep=",")
    df["qLen"] = df["form"].map(len)

    lemmaPosForm = defaultdict(dict)

    ### construct a dictionary mapping { lemma : { set(features) : form } }
    for i,row in df['feats'].items():
        #you could vectorize this if you hate row iters
        feats = []
        row = row[1:-1]
        pattern = r"\s"
        pattern1 = r"\'"
        row = re.sub(pattern, '', str(row))
        row = re.sub(pattern1, '', str(row))
        row = re.split(',', row)
        # print(row)

        for item in row:
            element = re.split(':', item)
            
            feats.append(element[-1])
        # print(feats)
        # input()
        # input(feats)

        # feats_new = []
        # for feats_num in range(len(feats)):
        #     for feat_item in feats[feats_num]:
        #         feats_new.append(feat_item)
        # input(feats_new)
        
        target = df.loc[i,"form"]
        lemma = df.loc[i,"lemma"]
        pos = frozenset(eval(str(feats)))
        # lemmaPosForm[lemma][pos] = target
        if lemma != "be":
            lemmaPosForm[lemma][pos] = target
    # for inp, block in df.groupby("INPUT"):
    #     inp = eval(inp)
    #     if "_" in inp:
    #         continue #why the heck is _ treated a word?
    #     idx = block["qLen"].idxmax()
    #     target = block.loc[[idx,], ["RESPONSE_FORM",]].values[0][0]
    #     lemma = inp[0]
    #     feats = inp[1:]
    #     #English: remove "being" from set
    #     if lemma != "be":
    #         lemmaPosForm[lemma][frozenset(feats)] = target
            # lemmaPosForm[lemma][feats] = target


    # find set of forms which are predictably identical across all lemmas
    print("lemmaPosForm",lemmaPosForm)
    
    cells = collapseSyncretism(lemmaPosForm)
    print("cells",cells)
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
            for i in vals:
                lis=[]
                for j in i:
                    lis.append(j)
                
                for indx, row_ in df['feats'].items():
                    feats_ = []
                    row_ = row_[1:-1]
                    pattern = r"\s"
                    pattern1 = r"\'"
                    row_ = re.sub(pattern, '', str(row_))
                    row_ = re.sub(pattern1, '', str(row_))
                    row_ = re.split(',', row_)
                    # print(row)

                    for item_ in row_:
                        element_ = re.split(':', item_)
                        feats_.append(element_[-1])
                    # print(feats)
                    # input()
                    # input(feats)

                            
                    """lis_row=[]
                    row =row.split('[')[-1]
                    row =row.split(']')[0]
                    lis_row=[]
                    for k in row.split(','):
                        k = re.sub(re.compile(r"\'"),'',k)
                        # print('\'',i)
                        
                        k = re.sub(re.compile(r"\s+"),'',k)
                        # print('space',i)
                        # input()
                        lis_row.append(k)"""
                    # print('lis_row115',lis_row)
                    # if lis_row==['Sing', 'Ger']:
                    #         print(row)
                    #         print(cellNames[cell])
                    #         input()
                        # input()
                    # if lis.sort() == ['Ind', 'Sing', '3', 'Past', 'Fin'].sort():
                    #         print('val',lis)
                    #         print('row',lis_row)
                    #         print(df.loc[indx,'QUERY'])
                    #         print('cellNames[cell]',cellNames[cell])
                    #         input()

                    if lis.sort()==feats_.sort():
                        same_lis=[]
                        val=cellNames[cell]
                        for p in val:
                            same_lis.append(p)
                        same_lis=str(same_lis)
                        df.loc[indx,'feats']=same_lis
                        # print('val',lis)
                        # print('row',feats_)
                        # print(df.loc[indx,'feats'])
                        


                    if len(feats_)>1:
                        """if lis_row == df.loc[indx,'RESPONSE_LEMMA']:
                            first=lis_row[0]
                            lis_row=lis_row[1:]
                            
                        # print('row',row)
                        # print('lis_row117',lis_row)
                        # input()
                            if lis.sort()==lis_row.sort():
                                
                                val_lis=[first]
                                # print(vals, "belong to", cellNames[cell])
                                val=cellNames[cell]
                                for p in val:
                                    val_lis.append(p)
                                val_lis=str(val_lis)
                                df.loc[indx,'QUERY']=val_lis
                                print('val',lis)
                                print('row',lis_row)
                                print(df.loc[indx,'QUERY'])
                        
                                # print("row",row)
                                # print('df.loc[indx',df.loc[indx,'QUERY'])
                                # input()
                    else:"""
                        if lis.sort()==feats_.sort():
                            val_lis=[]
                            # print(vals, "belong to", cellNames[cell])
                            val=cellNames[cell]
                            for p in val:
                                val_lis.append(p)
                            val_lis=str(val_lis)
                            df.loc[indx,'feats']=val_lis
                            # print('val',lis)
                            # print('row',feats_)
                            # print(df.loc[indx,'feats'])
                        
            df.to_csv(data_path / "encoded_df.csv", index=False)
            
