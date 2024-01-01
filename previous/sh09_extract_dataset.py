from pathlib import Path
import pandas as pd
import json
import re

data_path = Path("/users/PAS2062/delijingyic/project/morph/UD_Spanish-GSD")
dic = {}
data = pd.read_csv(
    data_path / "spanish_train.csv",
    dtype=str,
)
data['frequency'] = 0
for i, row in data['form'].iteritems():
    if row in dic.keys():
        dic[row] += 1
    else:
        dic[row] = 1

for i, row in data['form'].iteritems():
    data.loc[i, 'frequency'] = dic[row]
data = data.drop_duplicates(['form'])
data.to_csv(data_path / "spanish_train_freq.csv", index=False)
