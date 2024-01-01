import numpy as np
from os.path import join
from pathlib import Path
import pandas as pd
import re
from random import choice


class DataGenerator:

    def __init__(
        self,
        file_path: Path,
    ):
        super(__class__, self).__init__()
        self.file_path = file_path
        self.data_path = traindata_path


    # def freq(self, rt):
    #     data_path = Path("/users/PAS2062/delijingyic/project/morph/rl/dataset")
    #     elp = pd.read_csv(
    #         data_path / "data.csv",
    #         dtype=str,
    #     )
    #     sublex = pd.read_csv(
    #         data_path / "SUBTLEXusfrequencyabove1.csv",
    #         dtype=str,
    #     )

    #     for i, row in elp['Word'].items():
    #         for j, rowj in sublex["Word"].items():
    #             if row == rowj:
    #                 elp.loc[i, 'Log_Freq_HAL'] = sublex.loc[j, 'Lg10WF']

    #     elp.to_csv(data_path / "elp_withsublex.csv",
    #                index=False)

    #     elp = elp[elp["I_Mean_RT"] != ""]
    #     elp["I_Mean_RT"] = elp["I_Mean_RT"].apply(
    #         lambda x: float(x) if type(x) == str else x)
    #     elp["Log_Freq_HAL"] = elp["Log_Freq_HAL"].apply(
    #         lambda x: float(x) if type(x) == str else x)
    #     # y1=elp[elp["Log_Freq_HAL"]]

    #     # y1_std=np.std(y1["I_Mean_RT"])
    #     x = elp["Log_Freq_HAL"]
    #     y = elp["I_Mean_RT"]

    #     reverse_z = np.polyfit(elp["I_Mean_RT"], elp['Log_Freq_HAL'], 1)

    #     reverse_a = np.poly1d(reverse_z, r=False, variable=["x"])
    #     # yt=a(x)
    #     # mean_yt=np.mean(yt)
    #     # std_yt=np.std(yt)
    #     # pridict_y

    #     predict_x = reverse_a(rt)
    #     return predict_x

    def response_time(self, af):
        elp=pd.read_csv(
            self.data_path / "elp_withsublex.csv",
            dtype=str,
        )
        elp = elp[elp["I_Mean_RT"] != ""]
        elp["I_Mean_RT"] = elp["I_Mean_RT"].apply(
            lambda x: float(x) if type(x) == str else x)
        elp["Log_Freq_HAL"] = elp["Log_Freq_HAL"].apply(
            lambda x: float(x) if type(x) == str else x)
        # y1=elp[elp["Log_Freq_HAL"]]

        y1_std = np.std(elp["I_Mean_RT"])
        x = elp["Log_Freq_HAL"]
        y = elp["I_Mean_RT"]
        z = np.polyfit(elp["Log_Freq_HAL"], elp['I_Mean_RT'], 1)

        a = np.poly1d(z, r=False, variable=["x"])

        # yt=a(x)
        # mean_yt=np.mean(yt)
        # std_yt=np.std(yt)
        # pridict_y
        pridict_y = a(af)

        return pridict_y, y1_std

    def powerset(self, s):
        x = len(s)
        lis = []
        for i in range(1, 1 << x):
            lis.append([s[j] for j in range(x) if (i & (1 << j))])
        return lis

    def __getitem__(self, l, f):
        data = pd.read_csv(
            self.file_path,
        )
        data= data.dropna(axis=1, how='any')
        lemma = data['lemma']
        feat = data['feats']
        feat_lis = [
            'Mood', 'Number', 'Person', 'Tense', 'VerbForm', 'Gender', 'Voice'
        ]
        target_lemma = l
        target_feat = f
        target_feats = []
        target_feat = target_feat[1:-1]
        pattern = r"\s"
        pattern1 = r"\'"
        target_feat = re.sub(pattern, '', str(target_feat))
        target_feat = re.sub(pattern1, '', str(target_feat))
        target_feat = re.split(',', target_feat)
        # print(row)

        for tar_item in target_feat:
            tar_element = re.split(':', tar_item)
            target_feats.append(tar_element[-1])
        # input(target_feats)
        

        data_feat = data['feats']
        # print(data_feat)
        # input()
        output_feat = pd.DataFrame(columns={
            "form": '',
            "lemma": '',
            "upos": '',
            "feats": '',
            "frequency": '',
        })
        output_feat_lemma = pd.DataFrame(columns={
            "form": '',
            "lemma": '',
            "upos": '',
            "feats": '',
            "frequency": '',
        })

        lis = [target_lemma]
        
        for i in target_feats:
            # feat_i = target_feat[i]
            lis.append(i)
        # input(lis)
        pset = self.powerset(lis)
        # print(pset)
        # input()
        df_list = []
        # input(pset)
        for i in range(len(pset)):
            df = pd.DataFrame()
            for k, row in data['feats'].items():
                # if 'Typo' in row:
                #     continue
                # if 'Abbr' in row:
                #     continue
                # input(row)
                # input(feats_new)
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
                # print(feats)
                # input()
                # input(feats)

                feats_new = []
                for feats_num in range(len(feats)):
                    for feat_item in feats[feats_num]:
                        feats_new.append(feat_item)
                feats_new.append(data.loc[k, 'lemma'])

                # for j in pset[i]:
                # print(feats_new)
                new_row_pset = pd.DataFrame(data.loc[k])
                new_row_pset = new_row_pset.transpose()
                # print(new_row_pset)
                # input()
                if 'Typo' in feats_new:
                    continue
                elif 'Abbr' in feats_new:
                    continue
                elif (new_row_pset.empty == False):

                    # input(new_row_pset)

                    if (set(pset[i]) <= set(feats_new)):
                        # input(f"1, {new_row_pset}")

                        df= pd.concat([df, new_row_pset])
                    # input(df)
                    # print(df_list.append(df))
            if df.empty == True:
                continue
            else:
                df_list.append(df)
        # input(df_list)

        query_df = pd.DataFrame(columns={
            "INPUT": '',
            "QUERY": '',
            "ACTUAL_FEATURES": '',
            "RESPONSE_LEMMA": '',
            "RESPONSE_FORM": '',
            "TIME": '',
        })
        empty_data = {'form': ['empty'], 'lemma': ['empty'], 'upos': ['empty'], 'feats': [
            'empty'], 'frequency': [0], 'Log_Freq_HAL': [0], 'I_Mean_RT': [0]}
        df_empty = pd.DataFrame(empty_data)
        
        for i in range(len(df_list)):
            if df_list[i].empty == True:
                # print(pset[i])
                # exit(0)
                df_list[i] = df_empty

            # else:
            #     print(df_list[i])
            #     input()
        # print(len(df_list_new))
        # exit(0)
        # print(len(df_list))
        # exit(0)
        # print(len(pset))
        # print(df_list)
        # input()
        for i in range(len(df_list)):

            df_sing = pd.DataFrame(df_list[i])

            df_sing = df_sing.drop_duplicates(subset=['form', 'feats'])
            df_sing = df_sing.reset_index(drop=False)

            # df_sing=df_sing.set_index('Log_Freq_HAL')
            # df_sing=df_sing.reset_index(drop=False)

            # print(df_sing['Log_Freq_HAL'].sum())
            # input()
            log_freq = df_sing['Log_Freq_HAL'].sum()

            mean_rt_predict, y1_std = self.response_time(log_freq)
            # print(mean_rt_predict)

            item_rt = np.random.normal(mean_rt_predict, y1_std)

            low_bound_log_freq = data['Log_Freq_HAL'].sum()
            low_bound_rt, low_bound_std = self.response_time(
                low_bound_log_freq)
            item_rt = item_rt-low_bound_rt
            # input(low_bound_rt)
            # input(item_rt)
            # print('after',item_rt)
            # input()
            word = df_sing.sample(n=1)
            word = word.reset_index(drop=True)
            # print(word.loc[0,'form'])
            # exit(0)
            form = word['form'].loc[0]
            word_feat = word['feats'].loc[0]
            word_lemma = word['lemma'].loc[0]

            new_list = {"INPUT":[lis],"QUERY": [pset[i]], 'ACTUAL_FEATURES': [word_feat], 'RESPONSE_LEMMA': [
                word_lemma], 'RESPONSE_FORM': [form], 'TIME': [item_rt]}
            
            new_line = pd.DataFrame(new_list)
            query_df = pd.concat([query_df, new_line])
        
        df_clean=query_df.drop(query_df[query_df['ACTUAL_FEATURES']=='empty'].index)
        # print(df_clean)

        # for i in target_feat:
        #     feat_i = str(target_feat[i])
        #     for j, row in data['feats'].items():
        #         row = row[1:-1]
        #         pattern = r"\s"
        #         pattern1 = r"\'"
        #         row = re.sub(pattern, '', str(row))
        #         row = re.sub(pattern1, '', str(row))
        #         row = re.split(',', row)
        #         for item in row:

        #             element = re.split(':', item)

        #             # print(element)
        #             # # input()
        #             # print(feat_i)
        #             # input()
        #             new_row=pd.DataFrame(data.loc[j])
        #             new_row=new_row.transpose()
        #             if (feat_i in element):

        #                 output_feat = pd.concat([output_feat,new_row])
        #             # print(data.loc[j, "lemma"])
        #             # input()
        #             cur_lemma = data.loc[j, "lemma"]

        #             if (target_lemma == cur_lemma) and (feat_i in element):
        #                 output_feat_lemma = pd.concat([output_feat_lemma, new_row])
        return df_clean

        #     data_feat= data[()]
        # wordList_feat = data[]


if __name__ == '__main__':
    file_path = Path(
        "/users/PAS2062/delijingyic/project/morph/rl/dataset/new_ud_test_filter.csv")
    data_path = Path("/users/PAS2062/delijingyic/project/morph/rl/sq_output")
    traindata_path = Path("/users/PAS2062/delijingyic/project/morph/rl/dataset")

    datagenerator = DataGenerator(file_path)
    inputdata = pd.read_csv(
            file_path,
        )
    all_df = pd.DataFrame(columns={
            "INPUT": '',
            "QUERY": '',
            "ACTUAL_FEATURES": '',
            "RESPONSE_LEMMA": '',
            "RESPONSE_FORM": '',
            "TIME": '',
        })
    for i, row in inputdata['lemma'].items():

        l = row
        f = inputdata['feats'].loc[i]
        query_df = datagenerator.__getitem__(l, f)
        # output_feat=output_feat.drop_duplicates(subset=['form','feats'])
        # output_feat_lemma=output_feat_lemma.drop_duplicates(subset=['form','feats'])
        all_df= pd.concat([all_df, query_df])
        print(query_df)
        print("#####",all_df)
        # input()

    all_df.to_csv(data_path / "query_df_test.csv", index=False)
    # output_feat_lemma.to_csv(data_path / "output_feat_lemma.csv", index=False)

    # print(output_feat)
    # print(output_feat_lemma)
