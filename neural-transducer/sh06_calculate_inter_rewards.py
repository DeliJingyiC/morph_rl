import numpy as np
from os.path import join
from pathlib import Path
import pandas as pd
import re
from random import choice
import pandas as pd
import math
import sklearn

test_file =Path(
    "/users/PAS2062/delijingyic/project/morph/neural-transducer/checkpoints/transformer/tagtransformer/sigmorphon17-task1-dropout0.3/english-high-.decode.test_rt.tsv"
)

test_file_data = pd.read_csv(test_file,dtype=str,
                   sep="\t")
# print(test_file_data)
# input()
new_feats=['prediction','reward','state']
df=pd.DataFrame(columns=new_feats)
reward=0
total_list=[]
word_lis=[]

lemma_lis=[]
for i, row in test_file_data['prediction'].items():
    new_list=[]
    if i ==0:
        lemma_j=row.split('.')[0]
        new_list.append(row)
        new_list.append(test_file_data.loc[i,'reward'])
        new_list.append(test_file_data.loc[i,'state'])
        word_lis.append(new_list)
    else:
        lemma=row.split('.')[0]
        # print(row)
        # print(lemma)
        # print(lemma_j)

        # input()
        if lemma == lemma_j:
            new_list.append(row)
            new_list.append(test_file_data.loc[i,'reward'])
            new_list.append(test_file_data.loc[i,'state'])
            word_lis.append(new_list)
            new_list=[]
        else:
            total_list.append(word_lis)
            word_lis=[]
            lemma_j=lemma
            new_list.append(row)
            new_list.append(test_file_data.loc[i,'reward'])
            new_list.append(test_file_data.loc[i,'state'])
            word_lis.append(new_list)
            new_list=[]
inter_list=[]

for i in total_list:
    # print(i)
    # input()
    num=len(i)-1
    decision=[]
    while(num>=0):
        # print("total_query",i)
        if num == len(i)-1:
            inter_list.append(i[num][0])
            inter_list.append(i[num][1])
            inter_list.append("inter_state")
            df.loc[len(df.index)]=i[num]
            df.loc[len(df.index)]=inter_list
            decision.append(i[num-1])
            decision.append(inter_list)
            # print("first",decision)
            # input()
            inter_list=[]
            num-=1
        else:
            inter_list.append(i[num][0])  
            # print(float(decision[0][1]))
            # print(float(decision[1][1]))
            # input()
            reward=max(float(decision[0][1]),float(decision[1][1]))
            inter_list.append(reward)
            inter_list.append('inter_state')
            decision=[]
            decision.append(i[num-1])
            decision.append(inter_list)
            # print('second',decision)
            # input()
            df.loc[len(df.index)]=i[num]
            df.loc[len(df.index)]=inter_list
            
            inter_list=[]
            num-=1
        # print(df)
        # input()
df.to_csv("/users/PAS2062/delijingyic/project/morph/neural-transducer/checkpoints/transformer/tagtransformer/sigmorphon17-task1-dropout0.3/english-high-.decode.test_rt_intermediate.tsv", sep='\t',index=False)