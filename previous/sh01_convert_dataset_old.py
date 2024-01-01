from pathlib import Path
import pandas as pd
import json

data_path = Path("/users/PAS2062/delijingyic/project/morph/dataset")
pos_mapping = {"NOUN": "N", "VERB": "V", "ADJ": "ADJ", "ADV": "ADV"}

data = pd.read_csv(
    data_path / "DeriNetRU-0.5.tsv",
    delimiter='\t',
    dtype=str,
    names=[
        "ID",
        "Language-specific ID",
        "Lemma",
        "POS",
        "Morphological features",
        "Morpheme segmentation",
        "Main parent ID",
        "Parent relation",
        "None1",
        "None2",
    ],
)

data_derive = data[pd.notnull(data['Main parent ID'])]
derv_dict = {}
counter = 0
with open(data_path / "DeriNetRU-0.5_new.tsv", 'w') as p:
    for i, row in data_derive.iterrows():
        base_id = row['Main parent ID']
        base_row = data[data["ID"] == base_id].iloc[0]

        parent_pos = base_row["POS"]
        child_pos = row["POS"]
        if parent_pos == child_pos:
            continue
        lis = [
            base_row["Lemma"], row["Lemma"],
            f"{pos_mapping[base_row['POS']]}.{pos_mapping[row['POS']]}"
        ]
        counter += 1
        p.write("\t".join(lis) + "\n")

        if pos_mapping[parent_pos] not in derv_dict:
            derv_dict[pos_mapping[parent_pos]] = {pos_mapping[child_pos]: 1}
        else:
            if pos_mapping[child_pos] not in derv_dict[
                    pos_mapping[parent_pos]]:
                derv_dict[pos_mapping[parent_pos]][pos_mapping[child_pos]] = 1
            else:
                derv_dict[pos_mapping[parent_pos]][pos_mapping[child_pos]] += 1

with open(data_path / "data_stat.json", 'w') as f:
    json.dump(derv_dict, f)

print("new_dataset_length:", counter)
print("old_dataset_length:", len(data))
