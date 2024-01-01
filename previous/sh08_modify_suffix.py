from pathlib import Path
import pandas as pd
import json
import numpy as np

data_path = Path("/users/PAS2062/delijingyic/project/morph/dataset")
data = pd.read_csv(
    data_path / "DeriNetRU-0.5_updated.tsv",
    delimiter='\t',
    dtype=str,
)


modify ={
    "ADJ":{
    'в': 'ов',
    'вск': 'овск',
    'аск': 'ск',
    'ьск': 'ск',
    'ьн': 'н',
    'йн': 'н',
    'нн': 'енн',
    'нник': 'енник',
    'стическ':'истическ',
    'стск': 'истск'},

    "V":{
    'аив': 'ива',
    'изов': 'изова',
    'иров': 'ирова',
    'ров': 'ирова',
    'ачив': 'ачива',
    'ану': 'ну',
    },

    "N":{
    'т': 'от',
    'тк': 'отка',
    'тн': 'отн',
    'тник': 'отник',
    'новк': 'новка',
    'нт': 'ент',
    'ст': 'ист',
    'ьник': 'ник',
    'зм': 'изм',
    }
}

for i, row in data["btd"].iteritems():
    POS = data.loc[i,"POS"]
    base_pos=POS.split(".")[0]
    derive_pos=POS.split(".")[-1]
    if base_pos in modify and row in modify[base_pos]:
        # print(base_pos)
        # print(row)
        # print(modify[base_pos][row])
        # exit(0)
        data.loc[i,"btd"]=modify[base_pos][row]
        # print(data.loc[i,"btd"])
        # exit(0)
    elif derive_pos in modify and data.loc[i,'dtb'] in modify[derive_pos]:
        data.loc[i,"dtb"]=modify[derive_pos][data.loc[i,'dtb']]
    elif base_pos=="V" and derive_pos == "N":
        if data.loc[i,"btd"] == "арив" and data.loc[i,"dtb"] =="р":
            data.loc[i,"btd"] = "ива"
            data.loc[i,"dtb"] = ""
        elif data.loc[i,"btd"] == "ас" and data.loc[i,"dtb"] =="сл":
            data.loc[i,"btd"] = ""
            data.loc[i,"dtb"] = "л"
        elif data.loc[i,"btd"] == "ас" and data.loc[i,"dtb"] =="сток":
            data.loc[i,"btd"] = ""
            data.loc[i,"dtb"] = "ок"
        elif data.loc[i,"btd"] == "а" and data.loc[i,"dtb"] =="":
            data.loc[i,"btd"] = ""
            data.loc[i,"dtb"] = ""

for i, row in data["btd"].iteritems():
    POS = data.loc[i,"POS"]
    base_pos=POS.split(".")[0]
    derive_pos=POS.split(".")[-1]
    if base_pos == "N" and data.loc[i,"btd"] == "к":
        org_base=str(data.loc[i,"org_base"])
        if org_base.endswith("ка"):
            data.loc[i,"btd"] = "ка"
        elif data.loc[i,"org_derived"].endswith("ка"):
            data.loc[i,"dtb"] = "ка"
    elif base_pos == "V" and data.loc[i,"btd"] == "в":
        if data.loc[i,"org_base"].find("ивать")!=-1:
            # print(i)
            # print("org_base",data.loc[i,"org_base"])
            # print("btd",data.loc[i,"btd"])
            data.loc[i,"btd"] = "ива"
            # print("btd",data.loc[i,"btd"])

        else:
            data.loc[i,"btd"] = "ова"
    elif derive_pos == "V" and data.loc[i,"dtb"] == "в":
        if data.loc[i,"org_derived"].find("ивать")!=-1:
            data.loc[i,"dtb"] = "ива"      
        else:
            data.loc[i,"dtb"] = "ова"  
    
# Verb suffix арив and noun suffix р  ива and null, respectively
# Verb suffix ас and noun suffix сл  null and л, respectively
# Verb suffix ас and noun suffix сток  null and ок, respectively
# Verb suffix а and noun suffix null  both null

# Verb suffix в  ова UNLESS verb contains string ивать, in which case в  ива

# Noun suffix к  ка if noun ends in ка


data.to_csv(
    data_path / f"DeriNetRU-0.5_modified.tsv",
    sep='\t',
    index=False,
)