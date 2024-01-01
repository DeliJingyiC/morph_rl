import numpy as np
from os.path import join
from pathlib import Path
import pandas as pd
import re
from random import choice
import pandas as pd
import math
import sklearn
file_path = Path(
    "/users/PAS2062/delijingyic/project/morph/neural-transducer/encoded_df_dev.csv"
)
data = pd.read_csv(file_path,dtype=str,
                   sep=",",names=['lemma','word','feat'])
new_feats=['lemma','word','feat']
data = data.dropna(axis=0, how='any')
# data=sklearn.utils.shuffle(data)
df=pd.DataFrame(columns=new_feats)
for i, row in data['feat'].items():
    new_line=[]
    # print(data.loc[i, 'lemma'])
    # print(data.loc[i, 'lemma'].split(':')[0].split('.')[1].split(';')[0])
    removeinf=data.loc[i, 'lemma'].split(':')[0].split('.')[1].split(';')[0]
    if removeinf =='Inf':
        continue
    else:
        if len(data.loc[i, 'lemma'])>500:
            continue
        else:
            new_line.append(data.loc[i, 'lemma'].split(".")[0]+';'+data.loc[i, 'lemma'])
            # print(new_line)
            # input()
            new_line.append(data.loc[i, 'word'])
            new_line.append('no')
            df.loc[len(df.index)]=new_line
# print(len(data['word']))
        
    # data.loc[i, 'feat'] = data.loc[i, 'lemma'] 
    # data.loc[i, 'feat'] = "no" 
    # data.loc[i, 'lemma'] =data.loc[i, 'lemma'].split(".")[0]+';'+data.loc[i, 'lemma']
# print(len(data['word']))
# train_data=df.iloc[0:math.ceil(0.8*len(df['word']))]
# dev_data=df.iloc[math.ceil(0.8*len(df['word'])):math.ceil(0.9*len(df['word']))]
# test_data=df.iloc[math.ceil(0.9*len(df['word'])):]

df.to_csv("/users/PAS2062/delijingyic/project/morph/neural-transducer/dev_dataset.csv", header=None, sep='\t',index=False)
# dev_data.to_csv("/users/PAS2062/delijingyic/project/morph/neural-transducer/dev_data.csv", header=None, sep='\t',index=False)
# test_data.to_csv("/users/PAS2062/delijingyic/project/morph/neural-transducer/test_data.csv", header=None, sep='\t',index=False)

# feat_list = ["filename", "text"]
# df = pd.DataFrame(columns=feat_list)
# data_list = []
# for r,row in data["word"].items():
#     print(row)
#     input()
    # data_list.append(x)
# for x in range(len(data_list)):
#     ele = data_list[x].split("|")
#     df.loc[x, 'filename'] = ele[0]
#     df[x, 'text'] = ele[-1]
# print(df)
# df.to_csv(
    # "/users/PAS2062/delijingyic/project/wavegrad2/dataset/LJSpeech/LJSpeech-1.1/inference_text.csv",
    # index=False,
    # header=None,
    # sep='\t')
# lemma = data['lemma']
# feat = data['feats']
# feat_lis = ['Mood', 'Number', 'Person', 'Tense', 'VerbForm', 'Gender', 'Voice']
# dic = {}
# form_lis = []
# feat_list_1 = ['word', 'lemma', 'I_Mean_RT', 'Log_Freq_HAL', 'feats']
