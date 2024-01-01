import numpy as np
from os.path import join
from pathlib import Path
import pandas as pd
import re


class DataGenerator:

    def __init__(
        self,
        file_path: Path,
    ):
        super(__class__, self).__init__()
        self.file_path = file_path

    def __getitem__(self, l, f):
        data = pd.read_csv(
            self.file_path,
            names=['form', 'lemma', 'upos', 'feats', 'frequency'])

        lemma = data['lemma']
        feat = data['feats']
        feat_lis = [
            'Mood', 'Number', 'Person', 'Tense', 'VerbForm', 'Gender', 'Voice'
        ]
        target_lemma = l
        target_feat = f
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

        for i in target_feat:
            feat_i = str(target_feat[i])
            for j, row in data['feats'].iteritems():
                row = row[1:-1]
                pattern = r"\s"
                pattern1 = r"\'"
                row = re.sub(pattern, '', str(row))
                row = re.sub(pattern1, '', str(row))
                row = re.split(',', row)
                for item in row:

                    element = re.split(':', item)

                    # print(element)
                    # input()
                    # print(feat_i)
                    # input()

                    if (feat_i in element):
                        output_feat = output_feat.append(data.iloc[j])
                    # print(data.loc[j, "lemma"])
                    # input()
                    cur_lemma = data.loc[j, "lemma"]

                    if (target_lemma == cur_lemma) and (feat_i in element):
                        output_feat_lemma = output_feat_lemma.append(
                            data.iloc[j])
        return output_feat, output_feat_lemma

        #     data_feat= data[()]
        # wordList_feat = data[]


if __name__ == '__main__':

    l = "unite"
    f = {'Tense': 'Past', 'VerbForm': 'Part'}
    file_path = Path(
        "/users/PAS2062/delijingyic/project/morph/previous/english_train_freq.csv")
    data_path = Path("/users/PAS2062/delijingyic/project/morph/training_data")
    datagenerator = DataGenerator(file_path)
    output_feat, output_feat_lemma = datagenerator.__getitem__(l, f)
    output_feat.to_csv(data_path / "output_feat.csv", index=False)
    output_feat_lemma.to_csv(data_path / "output_feat_lemma.csv", index=False)

    # print(output_feat)
    # print(output_feat_lemma)
