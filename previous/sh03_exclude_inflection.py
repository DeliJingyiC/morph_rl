from pathlib import Path
import pandas as pd
import json

data_path = Path("/users/PAS2062/delijingyic/project/morph/dataset")
inflectional_marker = {
    "N":
    ["а", "я", "ё", "е", "о", "ы", "ь", "ья", "ье", "и", "ой", "ий", "ый"],
    "ADJ": ["ый", "ий", "ой", "ь"],
    "V": [
        "иться", "еться", "аться", "ться", "ить", "еть", "ать", "ть", "ти",
        "ять"
    ]
}

data = pd.read_csv(
    data_path / "DeriNetRU-0.5_new.tsv",
    delimiter='\t',
    dtype=str,
    names=[
        "base",
        "derived",
        "POS",
    ],
)
data['base_POS'] = 0
data['derived_POS'] = 0
data['org_base'] = data['base']
data['org_derived'] = data['derived']

# data['base_noinflex']=data['base']
# data['derived_noinflex']=data['derived']

for i, row in data['POS'].iteritems():

    data.loc[i, 'base_POS'] = row.split(".")[0]
    data.loc[i, 'derived_POS'] = row.split(".")[1]

for i, row in data['base_POS'].iteritems():
    if row == "N":
        if data.loc[i, 'base'][-2:] in inflectional_marker["N"]:
            data.loc[i, 'base'] = data.loc[i, 'base'][:-2]
        elif data.loc[i, 'base'][-1] in inflectional_marker["N"]:
            data.loc[i, 'base'] = data.loc[i, 'base'][:-1]
        else:
            continue

    if row == "ADJ":
        if data.loc[i, 'base'][-3:] in inflectional_marker["ADJ"]:
            data.loc[i, 'base'] = data.loc[i, 'base'][:-3]
        elif data.loc[i, 'base'][-2:] in inflectional_marker["ADJ"]:
            data.loc[i, 'base'] = data.loc[i, 'base'][:-2]
        else:
            continue

    if row == "V":
        if data.loc[i, 'base'][-5:] in inflectional_marker["V"]:
            data.loc[i, 'base'] = data.loc[i, 'base'][:-5]
        elif data.loc[i, 'base'][-4:] in inflectional_marker["V"]:
            data.loc[i, 'base'] = data.loc[i, 'base'][:-4]
        elif data.loc[i, 'base'][-3:] in inflectional_marker["V"]:
            data.loc[i, 'base'] = data.loc[i, 'base'][:-3]
        elif data.loc[i, 'base'][-2:] in inflectional_marker["V"]:
            data.loc[i, 'base'] = data.loc[i, 'base'][:-2]
        else:
            continue

for i, row in data['derived_POS'].iteritems():
    if row == "N":
        if data.loc[i, 'derived'][-2:] in inflectional_marker["N"]:
            data.loc[i, 'derived'] = data.loc[i, 'derived'][:-2]
        elif data.loc[i, 'derived'][-1] in inflectional_marker["N"]:
            data.loc[i, 'derived'] = data.loc[i, 'derived'][:-1]
        else:
            continue

    if row == "ADJ":
        if data.loc[i, 'derived'][-3:] in inflectional_marker["ADJ"]:
            data.loc[i, 'derived'] = data.loc[i, 'derived'][:-3]
        elif data.loc[i, 'derived'][-2:] in inflectional_marker["ADJ"]:
            data.loc[i, 'derived'] = data.loc[i, 'derived'][:-2]
        else:
            continue

    if row == "V":
        if data.loc[i, 'derived'][-5:] in inflectional_marker["V"]:
            data.loc[i, 'derived'] = data.loc[i, 'derived'][:-5]
        elif data.loc[i, 'derived'][-4:] in inflectional_marker["V"]:
            data.loc[i, 'derived'] = data.loc[i, 'derived'][:-4]
        elif data.loc[i, 'derived'][-3:] in inflectional_marker["V"]:
            data.loc[i, 'derived'] = data.loc[i, 'derived'][:-3]
        elif data.loc[i, 'derived'][-2:] in inflectional_marker["V"]:
            data.loc[i, 'derived'] = data.loc[i, 'derived'][:-2]
        else:
            continue
data = data.loc[:, ["base", "derived", "POS", "org_base", "org_derived"]]
count = 0
count_der = 0
count_both = 0
count_both_l = 0
for i, row in data['base'].iteritems():
    if len(row) > 0:
        continue
    elif len(row) == 0 and len(data.loc[i, "derived"]) == 0:
        data = data.drop(labels=i)
        count_both_l += 1
    else:
        data = data.drop(labels=i)
        count += 1
for i, row in data['derived'].iteritems():
    if len(row) > 0:
        continue
    elif len(row) == 0 and len(data.loc[i, "base"]) == 0:
        data = data.drop(labels=i)
        count_both += 1
    else:
        data = data.drop(labels=i)
        count_der += 1
print("count", count)
print("count_der", count_der)
print("count_both", count_both)
print("count_both+l", count_both_l)

data.to_csv(data_path / "DeriNetRU-0.5_noInflection.tsv",
            sep='\t',
            header=False,
            index=False)
