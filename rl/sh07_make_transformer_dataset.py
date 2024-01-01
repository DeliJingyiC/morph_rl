from pathlib import Path
import pandas as pd
import re
from random import choice
import pandas as pd 

# import sqlite3
data_path=Path("/users/PAS2062/delijingyic/project/morph/rl/dataset")
file_path = Path(
        "/users/PAS2062/delijingyic/project/morph/rl/dataset/new_ud_train_filter.csv")
data = pd.read_csv(file_path,)
data = data.dropna(axis=1, how='any')
lemma = data['lemma']
feat = data['feats']
# feat_lis = [
#             'Mood', 'Number', 'Person', 'Tense', 'VerbForm', 'Gender', 'Voice'
#         ]
dic={}
form_lis=[]
feat_list_1=['lemma','word','feats','rt']
# df=
# def mapping_df_types(df):

#     '''
#     db_info = {
#     'username':username ,
#     'passwords':passwords,
#     'server':server,
#     'port':port,
#     'database':database}

#     '''

#     dtypedict = {}
#     for i, j in zip(df.columns, df.dtypes):
#         if "object" in str(j):
#             dtypedict.update({i: NVARCHAR(length=255)})
#         if "float" in str(j):
#             dtypedict.update({i: Float(precision=2, asdecimal=True)})
#         if "int" in str(j):
#             dtypedict.update({i: Integer()})
#     return dtypedict
df=pd.DataFrame(columns=feat_list_1)
# for i, row in data['feats'].items():
#     feats = []
#     row = row[1:-1]
#     pattern = r"\s"
#     pattern1 = r"\'"
#     row = re.sub(pattern, '', str(row))
#     row = re.sub(pattern1, '', str(row))
#     row = re.split(',', row)
#     # print(row)

#     for item in row:
#         element = re.split(':', item)[-1]
#         # input(element)
#         feats.append(element)
#         # input(item)
#         # if element[-1] not in feat_list_1:
#         #     feat_list_1.append(element[-1])
        
    
#     # print(feats)
#     # input()
#     # input(feat_list_1)

    # feats_new = []
    # for feats_num in range(len(feats)):
    #     for feat_item in feats[feats_num]:
    #         feats_new.append(feat_item)
    # # data['feats_new'].loc[i]=feats_new
    # # feats_new.append(data.loc[i, 'lemma'])
    # # input(feats_new)
    # # input(data)

# feat_only=feat_list_1[5:]
#################################################################################
for i, row in data['feats'].items():
    strfeats=''
    # input(row)
    new_line=[]
    
    # new_line.append(data['lemma'].loc[i])
    # new_line.append(data['form'].loc[i])
    # new_line.append(data['I_Mean_RT'].loc[i])
    # new_line.append(data['Log_Freq_HAL'].loc[i])
    feats2 = []
    row = row[1:-1]
    pattern = r"\s"
    pattern1 = r"\'"
    row = re.sub(pattern, '', str(row))
    row = re.sub(pattern1, '', str(row))
    row = re.split(',', row)
    # print(row)

    for item in row:
        element = re.split(':', item)
        # input(element)
        if element[-1] == "Yes":
            continue
        feats2.append(element[-1])
    # input(feats2)
    if len(feats2) >1:
        for fe in feats2[:-1]:
            strfeats="V;"+fe + ";"
        # input(feats2[-1])
        # input(row)
        # input(strfeats)
        strfeats=strfeats + feats2[-1]
        # input(strfeats)
    else:
        for fe in feats2:
            strfeats="V;"+fe
    # data['feats'].loc[i]=strfeats
    # input(data['feats'].loc[i])

    strfeats=data['lemma'].loc[i] + ';'+ strfeats
    new_line.append(strfeats)
    new_line.append(data['form'].loc[i])
    new_line.append('no')
    new_line.append(data['I_Mean_RT'].loc[i])

    # input(feats2)
    # input(data['feats'].loc[i])
    # input(new_line)
    # for f in feat_only:
    #     if f not in feats2:
    #         new_line.append(0)
    #     else:
    #         new_line.append(1)
    # input(new_line)
    df.loc[len(df.index)]=new_line
# df=df.drop(df[df['Yes']>0].index)
# df=df.drop('Yes',axis=1)
# input(df)
df.to_csv(data_path / "english-train_withrt.csv",
            index=False,header=None, sep='\t')
