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
    "/users/PAS2062/delijingyic/project/morph/neural-transducer/english-uncovered-test.tsv"
)
test_result=Path("/users/PAS2062/delijingyic/project/morph/neural-transducer/checkpoints/transformer/tagtransformer/sigmorphon17-task1-dropout0.3/english-high-.decode.test.tsv")
test_rt=Path("/users/PAS2062/delijingyic/project/morph/rl/sq_output/query_df_test.csv")
test_file_data = pd.read_csv(test_file,dtype=str,
                   sep="\t", names=['prediction','target','zero'])

test_rt_data = pd.read_csv(test_rt,dtype=str,
                   sep=",")
wu_result_data = pd.read_csv(test_result,dtype=str,
                   sep="\t")
# print(wu_result_data)
# print(test_file_data)
# input()
# wu_result_data = wu_result_data.dropna(axis=0, how='any')
test_file_data['dist']=wu_result_data['dist']
# print(test_file_data)
# input()
test_max_idx=test_rt_data['TIME'].idxmax()
testmax=float(test_rt_data.loc[test_max_idx,'TIME'])
# print(testmax)
# input()

new_feats=['prediction','reward','state']
df=pd.DataFrame(columns=new_feats)
reward=0

# df_result=pd.DataFrame(columns=['prediction','target','loss','dist'])

# for i, row in wu_result_data['dist'].items():
#     new_line=[]

#     prediction=wu_result_data.loc[i,'prediction'].replace(' ','')
#     target=wu_result_data.loc[i,'target'].replace(' ','')
#     loss=wu_result_data.loc[i,'loss']
#     dist=int(wu_result_data.loc[i,'dist'])
    
#     new_line.append(prediction)
#     new_line.append(target)
#     new_line.append(loss)
#     new_line.append(dist)
#     df_result.loc[len(df.index)]=new_line


for i, row in test_file_data['prediction'].items():
    new_line=[]
    new_line.append(row)
    row= row.split(':')
    feature=row[-1]
    # print(feature)
    # input()
    inp=row[0]
    inp=inp.split(';')
    inp_string=inp[1]
    for item in inp[2:]:
        inp_string=inp_string+';'+item
    
    
    lemma=inp_string.split('.')[0]
    # print("inp",inp_string)
    # print("lemma",lemma)
    # input()
    if feature =='':
        continue
    else:
        feature=feature.split('>')[0].split('.')
        # print(feature)
        # input()
        if feature[0] != lemma:
            feature=feature[-1]
        else:
            feature_lemma=feature[0]
            # print("feature_lemma",feature_lemma)
            # input()
            for feat_i in feature[1:]:
                if feat_i != '':
                    feature_lemma = feature_lemma +';'+feat_i
                else:
                    continue
            feature = feature_lemma
        # print('feature',feature)
        # input()
    for j, row_j in test_rt_data['INPUT'].items():
        # print(row_j)
        # input()
        row_j=row_j.replace('\"','')
        # row_j=row_j.rstrip(']')
        # row_j=row_j.lstrip('[')
        row_j=eval(row_j)

        # lemma=row_j[0]
        feat_inp=row_j[0]+'.'+row_j[1]
        
        for row_j_i in row_j[2:]:
            feat_inp=feat_inp+';'+row_j_i
        if feat_inp==inp_string:
            # print("feat_inp",feat_inp)
            # input()
            query=eval(test_rt_data.loc[j,'QUERY'])
            
            if len(query)>1:
                query_string=query[0]
                for query_i in query[1:]:
                    query_string=query_string+';'+query_i
            else:
                query_string=query[0]
            # print("query_string",query_string)
            # input()

            if query_string ==feature:
                # print('query_string',query_string)
                # print('feature',feature)

                # input()
                dist=int(test_file_data.loc[i,'dist'])
                if dist==0:
                    reward=-float(test_rt_data.loc[j,'TIME'])
                else:
                    reward=-testmax
                    # print(i)
                    # print(test_file_data.loc[i,'prediction'])
                    # input()
                new_line.append(reward)
                new_line.append('stop_state')
                if len(new_line)==3:
                    df.loc[len(df.index)]=new_line
                else:
                    continue
                # print(new_line)
                # print(df)
                # input()
            # if 
        # print(feat_inp)
        # input()
df.to_csv("/users/PAS2062/delijingyic/project/morph/neural-transducer/checkpoints/transformer/tagtransformer/sigmorphon17-task1-dropout0.3/english-high-.decode.test_rt.tsv", sep='\t',index=False)

"""rt_file_dev = Path(
    "/users/PAS2062/delijingyic/project/morph/rl/dataset/new_ud_dev_filter.csv"
)

rt_file_test = Path(
    "/users/PAS2062/delijingyic/project/morph/rl/dataset/new_ud_test_filter.csv"
)

rt_file_train = Path(
    "/users/PAS2062/delijingyic/project/morph/rl/dataset/new_ud_train_filter.csv"
)

wu_results = Path(
    "/users/PAS2062/delijingyic/project/morph/neural-transducer/checkpoints/transformer/tagtransformer/sigmorphon17-task1-dropout0.3/english-high-.decode.test.tsv"
)
rt_data_dev = pd.read_csv(rt_file_dev,dtype=str,
                   sep=",")
rt_data_test = pd.read_csv(rt_file_test,dtype=str,
                   sep=",")
rt_data_train = pd.read_csv(rt_file_train,dtype=str,
                   sep=",")
wu_result_data = pd.read_csv(wu_results,dtype=str,
                   sep="\t")
wu_result_data = wu_result_data.dropna(axis=0, how='any')

new_feats=['prediction','target','loss','dist','reward']
df=pd.DataFrame(columns=new_feats)
# df=df.reset_index()
# df.drop('index',axis=1,inplace=True)
##find the lowest rt

dev_max_idx=rt_data_dev['I_Mean_RT'].idxmax()
devmax=float(rt_data_dev.loc[dev_max_idx,'I_Mean_RT'])
train_max_idx=rt_data_train['I_Mean_RT'].idxmax()
trainmax=float(rt_data_train.loc[train_max_idx,'I_Mean_RT'])
test_max_idx=rt_data_test['I_Mean_RT'].idxmax()
testmax=float(rt_data_test.loc[test_max_idx,'I_Mean_RT'])

for i, row in wu_result_data['dist'].items():
    new_line=[]

    prediction=wu_result_data.loc[i,'prediction'].replace(' ','')
    target=wu_result_data.loc[i,'target'].replace(' ','')
    loss=wu_result_data.loc[i,'loss']
    dist=int(wu_result_data.loc[i,'dist'])
    
    new_line.append(prediction)
    new_line.append(target)
    new_line.append(loss)
    new_line.append(dist)

    ##get rewards
    # if new_line==[]:
    #     continue
    # else:
    

    if dist == 0:
        index=rt_data_test[rt_data_test.form ==prediction].index.tolist()
        index=index[0]
        
        print(rt_data_test.loc[index,'I_Mean_RT'])
        # input()
        reward=-float(rt_data_test.loc[index,'I_Mean_RT'])
        # print("before",new_line)
        new_line.append(reward)
        print(new_line)
        # input()
        df.loc[len(df.index)]=new_line
        reward=0
        # new_line=[]

    else:
        # continue
        reward=-testmax
        new_line.append(reward)
        df.loc[len(df.index)]=new_line
        # new_line=[]
        reward=0
        # print(df)
    df.to_csv("/users/PAS2062/delijingyic/project/morph/neural-transducer/checkpoints/transformer/tagtransformer/sigmorphon17-task1-dropout0.3/english-high-.decode.test_rt.tsv", sep='\t',index=False)
# print(rt_data)
# print(wu_result_data)

# new_feats=['lemma','word','feat']
# data = data.dropna(axis=0, how='any')"""