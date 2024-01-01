import numpy as np
from os.path import join
from pathlib import Path
import pandas as pd
import re


traindata_path = Path("/users/PAS2062/delijingyic/project/morph/rl/sq_output")
query_df=pd.read_csv(
            traindata_path / "query_df.csv",
            dtype=str,
            # index_col=0
        )
# input(query_df)
for i,row in query_df['INPUT'].items():
    if 'Yes' in row:
        # input(row)
        # input(i)
        query_df=query_df.drop(index=i)
    # if 'Yes' in query_df['QUERY']:
    #     query_df=query_df.drop(query_df.index[i])

query_df.to_csv(traindata_path / "query_df.csv", index=False)