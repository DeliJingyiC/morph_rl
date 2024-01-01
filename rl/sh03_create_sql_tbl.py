import numpy as np
from os.path import join
from pathlib import Path
import pandas as pd
import re
from random import choice
# import pandas as pd 
from sqlalchemy import create_engine
from sqlalchemy.types import NVARCHAR, Float, Integer
# import sqlite3

file_path = Path(
        "/users/PAS2062/delijingyic/project/morph/rl/dataset/ud_train.csv")
data = pd.read_csv(file_path,)
data = data.dropna(axis=1, how='any')
lemma = data['lemma']
feat = data['feats']
feat_lis = [
            'Mood', 'Number', 'Person', 'Tense', 'VerbForm', 'Gender', 'Voice'
        ]
dic={}
form_lis=[]
feat_list_1=['word','lemma','I_Mean_RT','Log_Freq_HAL','feats']
# df=
def mapping_df_types(df):

    '''
    db_info = {
    'username':username ,
    'passwords':passwords,
    'server':server,
    'port':port,
    'database':database}

    '''

    dtypedict = {}
    for i, j in zip(df.columns, df.dtypes):
        if "object" in str(j):
            dtypedict.update({i: NVARCHAR(length=255)})
        if "float" in str(j):
            dtypedict.update({i: Float(precision=2, asdecimal=True)})
        if "int" in str(j):
            dtypedict.update({i: Integer()})
    return dtypedict

for i, row in data['feats'].items():
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
        feats.append(element)
        # input(item)
        if element[-1] not in feat_list_1:
            feat_list_1.append(element[-1])
        
    
    # print(feats)
    # input()
    # input(feat_list_1)

    feats_new = []
    for feats_num in range(len(feats)):
        for feat_item in feats[feats_num]:
            feats_new.append(feat_item)
    # data['feats_new'].loc[i]=feats_new
    # feats_new.append(data.loc[i, 'lemma'])
    # input(feats_new)
    # input(data)
df=pd.DataFrame(columns=feat_list_1)
feat_only=feat_list_1[5:]
#################################################################################
for i, row in data['feats'].items():
    new_line=[]
    new_line.append(data['form'].loc[i])
    new_line.append(data['lemma'].loc[i])
    new_line.append(data['I_Mean_RT'].loc[i])
    new_line.append(data['Log_Freq_HAL'].loc[i])
    new_line.append(data['feats'].loc[i])




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
        feats2.append(element[-1])
    for f in feat_only:
        if f not in feats2:
            new_line.append(0)
        else:
            new_line.append(1)
    # input(new_line)
    df.loc[len(df.index)]=new_line
df=df.drop(df[df['Yes']>0].index)
df=df.drop('Yes',axis=1)

engine=create_engine('sqlite:///foo.db')
# db_info = {
#     'username':"jyc" ,
#     'passwords':"jingyi",
#     'server':"“localhost",
#     'port':3306,
#     'database':"database"}
# engine = create_engine('mssql+pymssql://{}:{}@{}:{}/{}?charset=utf8'.format(db_info['username'],db_info['passwords'],db_info['server'],db_info['port'],db_info['database']))
conn = engine.connect() 
df.to_sql('feat_tbls',con=conn)

    # 关闭连接
conn.close()
# for item in all:
#     print(item.name, item.fullname)
# print(df)      
    
    # print(feats)
    # input()
    # input(feats)

    # feats_new2 = []
    # for feats_num in range(len(feats2)):
    #     for feat_item in feats2[feats_num]:
    #         feats_new2.append(feat_item)

# print(len(form_lis))
# print(feats_new)
# input()