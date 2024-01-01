from pathlib import Path
import pandas as pd


class DataGenerator:

    def __init__(
        self,
        # file_path: Path,
        # traindata_path: Path
    ):
        super(__class__, self).__init__()
        # self.file_path = file_path
        # self.data_path = traindata_path
    
    def convert(self,file_path,data_path):
        english_ud=pd.read_csv(
        file_path,
        dtype=str,
    )
        english_data=pd.read_csv(
        data_path,
        dtype=str,
    )
        for i, row in english_data['form'].items():
            for j, row_j in english_ud['form'].items():
                if row_j == row:
                    if english_ud.loc[j,'feats'] !=english_data.loc[i,'feats']:
                        new_list=[]
                        new_list.append(english_data.loc[i,'form'])
                        new_list.append(english_data.loc[i,'lemma'])
                        new_list.append(english_data.loc[i,'upos'])
                        new_list.append(english_ud.loc[j,'feats'])
                        new_list.append(english_data.loc[i,'frequency'])
                        new_list.append(english_data.loc[i,'Log_Freq_HAL'])
                        new_list.append(english_data.loc[i,'I_Mean_RT'])
                        # print(english_data.columns)
                        # input(new_list)
                        english_data.loc[len(english_data.index)]=new_list
        english_data=english_data.drop_duplicates()
        return english_data
if __name__ == '__main__':
       
    data_path_ud = Path("/users/PAS2062/delijingyic/project/morph/previous/UD_English-GUM")
    data_path_train = Path("/users/PAS2062/delijingyic/project/morph/rl/dataset")
    file_path=data_path_ud/"english_dev.csv"
    data_path=data_path_train/"ud_dev.csv"
    datageneraor=DataGenerator()
    english_data=datageneraor.convert(file_path,data_path)
    english_data.to_csv(data_path_train / "new_ud_dev.csv", index=False)
    