from pathlib import Path
import pandas as pd
import json
import numpy as np

data_path = Path("/users/PAS2062/delijingyic/project/morph/dataset")
data = pd.read_csv(
    data_path / "DeriNetRU-0.5_all.tsv",
    delimiter='\t',
    dtype=str,
)
data_all = pd.read_csv(
    data_path / "DeriNetRU-0.5_all.tsv",
    delimiter='\t',
    dtype=str,
)

reversal_special = {
    'н': 'в',
    'ат': 'в',
    'ск': 'ин',
    'к': 'н',
    'и': 'н',
    'ь': 'н',
}
data = data[data[["derivation_process_suffix",
                  "derivation_process_prefix"]].notna().any(axis=1)]
# data = data[data["derivation_process_prefix"].notna()]
# print(data)
# input()
# data['btd'] = 0
# data['dtb'] = 0
data["derivation_process_suffix"].fillna("", inplace=True)
#base suffix
data["btd"] = data["derivation_process_suffix"].apply(
    lambda x: x.split(".")[0])  
#derived suffix
data["dtb"] = data["derivation_process_suffix"].apply(
    lambda x: x.split(".")[-1])
# for i, row in data["derivation_process_suffix"].iteritems():
#     if len(row) > 0:
#         bTd = row.split(".")[0]
#         dTb = row.split(".")[-1]
#         data.loc[i, "btd"] = bTd
#         data.loc[i, "dtb"] = dTb

# data = data[(data["derivation_process_suffix"].notnull())
#             & (data["derivation_process_suffix"] != "")]
for i, row in data['derivation_process_suffix'].iteritems():
    if len(data.loc[i,'btd']) > 0 and len(data.loc[i,'dtb']) <= 0:
            # print("org_base", org_base)
            # print("org_derived", org_derived)
            # print("pos", pos)
            # print("derivation_process_prefix", derivation_process_prefix)
            # print('derivation_process_suffix', derivation_process_suffix)
            base_suffix = data.loc[i, 'btd']
            derived_suffix = data.loc[i, 'dtb']
            org_d=data.loc[i, 'org_derived']
            org_b=data.loc[i, 'org_base']
            der=data.loc[i, 'derived']
            bas=data.loc[i, 'base']
            pos_shift = data.loc[i, 'POS']

            data.loc[i, 'btd'] = derived_suffix
            data.loc[i, 'dtb'] = base_suffix
            data.loc[i, 'org_base'] = org_d
            data.loc[i, 'org_derived'] = org_b
            data.loc[i, 'base'] = der
            data.loc[i, 'derived'] = bas
            pos1 = pos_shift.split('.')[0]
            pos2 = pos_shift.split('.')[-1]

            if len(str(row)) > 0:
                derivation_process_prefix1 = str(
                    data.loc[i, 'derivation_process_prefix']).split('.')[0]
                derivation_process_prefix2 = str(
                    data.loc[i, 'derivation_process_prefix']).split('.')[-1]
            else:
                pass
            derivation_process_suffix1 = row.split(
                '.')[0]
            derivation_process_suffix2 = row.split(
                '.')[-1]
            data.loc[i, 'POS'] = pos2 + '.' + pos1
            data.loc[
                i,
                'derivation_process_prefix'] = derivation_process_prefix2 + '.' + derivation_process_prefix1
            data.loc[
                i,
                'derivation_process_suffix'] = derivation_process_suffix2 + '.' + derivation_process_suffix1
            # print(data.loc[ind, 'org_base'])
            # print(data.loc[ind, 'org_derived'])
            # print(data.loc[ind, 'POS'])
            # print(data.loc[ind, 'derivation_process_prefix'])
            # print(data.loc[ind, 'derivation_process_suffix'])

            # input()

groups = data[["btd", "dtb"]].groupby(["btd"])

total_index = []
for base in data["btd"].unique():
    derive_onBase = groups.get_group(base)["dtb"].unique()

    if (len(derive_onBase) <= 0):
        continue
    for derived in derive_onBase:
        try:
            # print("group1", groups.get_group(derived))
            # print("group2", groups.get_group(derived)["dtb"])
            # print("group3", groups.get_group(derived)["dtb"].unique())
            # input()
            drived_onDerived = groups.get_group(derived)["dtb"].unique()
            if (base in drived_onDerived.tolist()):
                if False and base == derived:
                    continue
                else:
                    total_index.extend(
                        data.query(
                            f"btd=='{base}' and dtb=='{derived}'").index)

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

# print(data.loc[total_index])
# input()
subdata = data.loc[total_index]
subdata_index = list(subdata.index)
new_index = []
repeat_index = []
while (len(subdata_index) > 0):
    print(f"\r{len(subdata_index)}", end="")
    ind = subdata_index[0]
    row = subdata.loc[ind]
    org_base = row['org_base']
    org_derived = row['org_derived']
    pos = row['POS']
    derivation_process_prefix = row['derivation_process_prefix']
    derivation_process_suffix = row['derivation_process_suffix']
    base = row["base"]
    derived = row["derived"]
    btd = row["btd"]
    dtb = row["dtb"]
    subdata_index.remove(ind)
    new_index.append(ind)

    try:
        rev_row = subdata.query(
            f"base=='{derived}' and derived=='{base}' and btd=='{dtb}' and dtb=='{btd}'"
        )
        # print(rev_row.index[0])
        # input()

        subdata_index.remove(rev_row.index[0])
        new_index.append(rev_row.index[0])
        # print(new_index)
        # input()
        # print(subdata.loc[[ind, rev_row.index[0]]])
        repeat_index.extend([ind, rev_row.index[0]])
        # print(repeat_index)
        # input()
    except:
        pass
for i in repeat_index:
    # print(i)
    # exit(0)
    dtb_spe = data.loc[i, 'dtb']
    btd_spe = data.loc[i, 'btd']
    org_base_spe = data.loc[i, 'org_base']
    org_derived_spc = data.loc[i, 'org_derived']
    pos_spc = data.loc[i, 'POS']
    derivation_process_prefix_spc = data.loc[i, 'derivation_process_prefix']
    derivation_process_suffix_spc = data.loc[i, 'derivation_process_suffix']
    base_spc = data.loc[i, 'base']
    derived_spc = data.loc[i, 'derived']
    # print("dtb_spe", dtb_spe)
    # print("btd_spe", btd_spe)
    # print("org_base_spe", org_base_spe)
    # print("org_derived_spc", org_derived_spc)
    # print("pos_spc", pos_spc)
    # print("derivation_process_prefix_spc",
    #       derivation_process_prefix_spc)
    # print("derivation_process_suffix_spc",
    #       derivation_process_suffix_spc)
    # print("base_spc", base_spc)
    # print("derived_spc", derived_spc)
    # exit(0)

    if dtb_spe in reversal_special and reversal_special[dtb_spe] == btd_spe:
        data.loc[i, 'btd'] = dtb_spe
        data.loc[i, 'dtb'] = btd_spe
        data.loc[i, 'org_base'] = org_derived_spc
        data.loc[i, 'org_derived'] = org_base_spe
        data.loc[i, 'base'] = derived_spc
        data.loc[i, 'derived'] = base_spc
        pos1 = pos_spc.split('.')[0]
        pos2 = pos_spc.split('.')[-1]
        if len(str(derivation_process_prefix_spc)) > 0:
            derivation_process_prefix1 = str(
                derivation_process_prefix_spc).split('.')[0]
            derivation_process_prefix2 = str(
                derivation_process_prefix_spc).split('.')[-1]
        else:
            pass
        derivation_process_suffix1 = dtb_spe
        derivation_process_suffix2 = dtb_spe
        data.loc[i, 'POS'] = pos2 + '.' + pos1
        data.loc[
            i,
            'derivation_process_prefix'] = derivation_process_prefix2 + '.' + derivation_process_prefix1
        data.loc[
            i,
            'derivation_process_suffix'] = derivation_process_suffix1 + '.' + derivation_process_suffix2



data.loc[repeat_index].to_csv(
    data_path / f"DeriNetRU-0.5_rule_reverse.tsv",
    sep='\t',
    index=False,
)

data.to_csv(
    data_path / f"DeriNetRU-0.5_updated.tsv",
    sep='\t',
    index=False,
)

    # print(data_path / "DeriNetRU-0.5_rule_reverse.tsv")
    # print(data_path / "DeriNetRU-0.5_updated.tsv")
