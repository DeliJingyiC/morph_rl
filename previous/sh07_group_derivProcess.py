from pathlib import Path
import pandas as pd
import json
import numpy as np

data_path = Path("/users/PAS2062/delijingyic/project/morph/dataset")
data = pd.read_csv(
    data_path / "DeriNetRU-0.5_deriveProcess.tsv",
    delimiter='\t',
    dtype=str,
)
data["derivation_process_suffix"].fillna("", inplace=True)
data["base_suffix"] = data["derivation_process_suffix"].apply(
    lambda x: x.split(".")[0]).apply(lambda x: x[max(0,
                                                     len(x) - 2):])
data["derivForm_suffix"] = data["derivation_process_suffix"].apply(
    lambda x: x.split(".")[-1]).apply(lambda x: x[max(0,
                                                      len(x) - 2):])
# groups_dtd = data.groupby('derivForm_suffix')
# groups_base_suffix = data.groupby('base_suffix')

data.index.name = "ID"
data.reset_index(inplace=True)
data_derivForm_suffix = data.set_index(
    ["derivForm_suffix", "base_suffix", "ID"])
data_derivForm_suffix.sort_index(inplace=True, ascending=False)
data_derivForm_suffix.to_csv(
    data_path / "DeriNetRU-sh07_derivForm_suffix.tsv",
    sep='\t',
    index=True,
)
print(data_derivForm_suffix)
# input()