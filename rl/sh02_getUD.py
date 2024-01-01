from pathlib import Path
import pandas as pd
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import optimize
data_path_ud = Path("/users/PAS2062/delijingyic/project/morph/previous")
data_path = Path("/users/PAS2062/delijingyic/project/morph/rl/dataset")
elp_withsublex=pd.read_csv(
    data_path /"elp_withsublex.csv",
    dtype=str,
)

elp_withsublex = elp_withsublex[elp_withsublex["I_Mean_RT"]!=""]
elp_withsublex["I_Mean_RT"] = elp_withsublex["I_Mean_RT"].apply(lambda x: float(x) if type(x)==str else x)
elp_withsublex["Log_Freq_HAL"] = elp_withsublex["Log_Freq_HAL"].apply(lambda x: float(x) if type(x)==str else x)

z = np.polyfit(elp_withsublex["Log_Freq_HAL"], elp_withsublex['I_Mean_RT'],1)

a= np.poly1d(z,r=False,variable=["x"])

ud_train=pd.read_csv(
    data_path_ud /"english_train_freq.csv",
    dtype=str,
)
ud_train['Log_Freq_HAL'] = 0
ud_train['I_Mean_RT'] = 0

ud_dev=pd.read_csv(
    data_path_ud /"english_dev_freq.csv",
    dtype=str,
)
ud_dev['Log_Freq_HAL'] = 0
ud_dev['I_Mean_RT'] = 0

ud_test=pd.read_csv(
    data_path_ud /"english_test_freq.csv",
    dtype=str,
)
ud_test['Log_Freq_HAL'] = 0
ud_test['I_Mean_RT'] = 0

for i, row in ud_train['form'].items():
    for j, rowj in elp_withsublex ['Word'].items():
        if row == rowj:
            ud_train.loc[i,'Log_Freq_HAL'] = elp_withsublex.loc[j,'Log_Freq_HAL']

for i, row in ud_dev['form'].items():
    for j, rowj in elp_withsublex ['Word'].items():
        if row == rowj:
            ud_dev.loc[i,'Log_Freq_HAL'] = elp_withsublex.loc[j,'Log_Freq_HAL']

for i, row in ud_test['form'].items():
    for j, rowj in elp_withsublex ['Word'].items():
        if row == rowj:
            ud_test.loc[i,'Log_Freq_HAL'] = elp_withsublex.loc[j,'Log_Freq_HAL']


pridict_y_ud_train=a(ud_train["Log_Freq_HAL"])
ud_train['I_Mean_RT'] = pridict_y_ud_train

pridict_y_ud_dev=a(ud_dev["Log_Freq_HAL"])
ud_dev['I_Mean_RT'] = pridict_y_ud_dev

pridict_y_ud_test=a(ud_test["Log_Freq_HAL"])
ud_test['I_Mean_RT'] = pridict_y_ud_test

ud_train.to_csv(data_path / "ud_train.csv",
            index=False)
ud_dev.to_csv(data_path / "ud_dev.csv",
            index=False)
ud_test.to_csv(data_path / "ud_test.csv",
            index=False)