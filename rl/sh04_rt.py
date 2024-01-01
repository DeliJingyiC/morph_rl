import numpy as np
from os.path import join
from pathlib import Path
import pandas as pd
import re
from random import choice
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

class DataGenerator:

    def __init__(
        self,
        file_path: Path,
    ):
        super(__class__, self).__init__()
        self.file_path = file_path
        self.data_path = traindata_path
        

    """def freq(self, rt):
        data_path = Path("/users/PAS2062/delijingyic/project/morph/rl/dataset")
        elp = pd.read_csv(
            data_path / "data.csv",
            dtype=str,
        )
        sublex = pd.read_csv(
            data_path / "SUBTLEXusfrequencyabove1.csv",
            dtype=str,
        )

        for i, row in elp['Word'].items():
            for j, rowj in sublex["Word"].items():
                if row == rowj:
                    elp.loc[i, 'Log_Freq_HAL'] = sublex.loc[j, 'Lg10WF']

        elp.to_csv(data_path / "elp_withsublex.csv",
                   index=False)

        elp = elp[elp["I_Mean_RT"] != ""]
        elp["I_Mean_RT"] = elp["I_Mean_RT"].apply(
            lambda x: float(x) if type(x) == str else x)
        elp["Log_Freq_HAL"] = elp["Log_Freq_HAL"].apply(
            lambda x: float(x) if type(x) == str else x)
        # y1=elp[elp["Log_Freq_HAL"]]

        # y1_std=np.std(y1["I_Mean_RT"])
        x = elp["Log_Freq_HAL"]
        y = elp["I_Mean_RT"]

        reverse_z = np.polyfit(elp["I_Mean_RT"], elp['Log_Freq_HAL'], 1)

        reverse_a = np.poly1d(reverse_z, r=False, variable=["x"])
        # yt=a(x)
        # mean_yt=np.mean(yt)
        # std_yt=np.std(yt)
        # pridict_y

        predict_x = reverse_a(rt)
        return predict_x
"""
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

    def __getitem__(self, sql):
        data = pd.read_csv(
            self.file_path,
        )
        engine=create_engine('sqlite:///foo.db')
        Session = sessionmaker(bind=engine)
        session = Session()
        conn = engine.connect() 
        query_df = pd.DataFrame(columns={
            "QUERY": '',
            # "ACTUAL_FEATURES": '',
            "RESPONSE_LEMMA": '',
            "RESPONSE_FORM": '',
            "TIME": '',
        })
        for j in sql:
            
            feats_tbl = pd.read_sql(j, conn)
            # input(j)
            # input(feats_tbl)
            # input(feats_tbl)
            # for i, row in feats_tbl['I_Mean_RT'].items():
            log_freq = feats_tbl['Log_Freq_HAL'].sum()

            mean_rt_predict, y1_std = self.response_time(log_freq)
            # print(mean_rt_predict)

            item_rt = np.random.normal(mean_rt_predict, y1_std)

            low_bound_log_freq = data['Log_Freq_HAL'].sum()
            low_bound_rt, low_bound_std = self.response_time(
                low_bound_log_freq)
            item_rt = item_rt-low_bound_rt
            word = feats_tbl.sample(n=1)
            
            word = word.reset_index(drop=True)
            # print(word.loc[0,'form'])
            # exit(0)
            form = word['word'].loc[0]
            word_feat = word['feats'].loc[0]
            word_lemma = word['lemma'].loc[0]

            new_list = {"QUERY": j, 'ACTUAL_FEATURES': [word_feat], 'RESPONSE_LEMMA': [word_lemma], 'RESPONSE_FORM': [form], 'TIME': [item_rt]}
            # new_list = {"QUERY": j, 'RESPONSE_LEMMA': [word_lemma], 'RESPONSE_FORM': [form], 'TIME': [item_rt]}
            new_line = pd.DataFrame(new_list)
            # input(new_line)

            query_df = pd.concat([query_df, new_line])
            # input(query_df)
                # input(item_rt)
            # input(feats_tbl.index.tolist())
            # r1 = session.query(feats_tbl).filter(feats_tbl.lemma == l,feats_tbl=)
        return(query_df)


        
if __name__ == '__main__':
    
    sql=["SELECT * FROM feat_tbls WHERE LEMMA='unite' OR PAST>0 OR PART>0","SELECT * FROM feat_tbls WHERE LEMMA='unite'","SELECT * FROM feat_tbls WHERE PAST>0","SELECT * FROM feat_tbls WHERE PART>0","SELECT * FROM feat_tbls WHERE LEMMA='unite' OR PAST>0 ","SELECT * FROM feat_tbls WHERE LEMMA='unite' OR PART>0","SELECT * FROM feat_tbls WHERE PAST>0 OR PART>0",]
    traindata_path = Path("/users/PAS2062/delijingyic/project/morph/rl/dataset")
    file_path = Path(
        "/users/PAS2062/delijingyic/project/morph/rl/dataset/ud_train.csv")
    data_path = Path("/users/PAS2062/delijingyic/project/morph/rl/sq_output")
    datagenerator = DataGenerator(file_path)
    query_df = datagenerator.__getitem__(sql)
    # output_feat=output_feat.drop_duplicates(subset=['form','feats'])
    # output_feat_lemma=output_feat_lemma.drop_duplicates(subset=['form','feats'])

    query_df.to_csv(data_path / "query_df.csv", index=False)
    # output_feat_lemma.to_csv(data_path / "output_feat_lemma.csv", index=False)

    # print(output_feat)
    # print(output_feat_lemma)
