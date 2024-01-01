from pathlib import Path
import pandas as pd
import json
import numpy as np

data_path = Path("/users/PAS2062/delijingyic/project/morph/dataset")
data = pd.read_csv(
    data_path / "DeriNetRU-0.5_deriveProcess.tsv",
    delimiter='\t',
    dtype=str,
    names=[
        "org_base", "org_derived", "POS",
        "derivation_prefix,deravation_suffix", "base", "derived"
    ],
)
print(data)

groups = data[["org_derived", "org_base"]].groupby(["org_base"])

total_index = []
for base in data["org_base"].unique():
    derive_onBase = groups.get_group(base)["org_derived"]
    if (len(derive_onBase) <= 0):
        continue
    for derived in derive_onBase:
        if base == derived:
            continue
        try:
            drived_onDerived = groups.get_group(derived)["org_derived"]
            if (base in drived_onDerived.tolist()):
                query = f"org_base=='{base}' and org_derived=='{derived}'"
                total_index.extend(data.query(query).index)

                # print(query)
                # print(data.query(query))
                # input()

                # print(f"reverse pair {base} {derived}")
            else:
                # print(f"{base} not found reversively in drived derived")
                # print(
                #     f"{base} not found in {derived} {drived_onDerived.tolist()}"
                # )
                # input()
                pass
        except:
            # print(f"{derived} not found reversively in {base} derived")
            pass

    # print(derive_onBase)
    # input()

data.iloc[total_index].to_csv(data_path / "DeriNetRU-0.5_reverses.tsv",
                              sep='\t',
                              index=False)
