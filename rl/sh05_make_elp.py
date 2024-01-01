import numpy as np
from os.path import join
from pathlib import Path
import pandas as pd

traindata_path = Path("/users/PAS2062/delijingyic/project/morph/rl/dataset")
elp = pd.read_csv(
            traindata_path / "data.csv",
            dtype=str,
        )
sublex = pd.read_csv(
            traindata_path / "SUBTLEXusfrequencyabove1.csv",
            dtype=str,
        )
for i, row in elp['Word'].items():
            for j, rowj in sublex["Word"].items():
                if row == rowj:
                    elp.loc[i, 'Log_Freq_HAL'] = sublex.loc[j, 'Lg10WF']

elp.to_csv(traindata_path / "elp_withsublex.csv",
                   index=False)