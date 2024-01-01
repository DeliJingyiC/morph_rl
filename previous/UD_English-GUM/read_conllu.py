from pyconll import load_from_file
from pathlib import Path
from conllu import parse_incr
from io import open
import pandas as pd

path = Path("/users/PAS2062/delijingyic/project/morph/UD_English-GUM")
# data = load_from_file(path / "en_gum-ud-dev.conllu")
# sentences = parse(data)
# print(sentences[0])
data_file = open(path / "en_gum-ud-train.conllu", "r", encoding="utf-8")
# print(data_file)
data = []
for tokenlist in parse_incr(data_file):
    for ids in tokenlist:
        data.append([
            ids["form"], ids["lemma"], ids["upos"], ids["xpos"], ids["feats"],
            ids["head"], ids["deprel"], ids["deps"], ids["misc"]
        ])
# data = str(data)
df = pd.DataFrame(data,
                  columns=[
                      'form', "lemma", "upos", "xpos", "feats", "head",
                      "deprel", "deps", "misc"
                  ])
df = df.loc[:, [
    "form",
    "lemma",
    "upos",
    "feats",
]]
form_index = list(df.columns).index('form')
for i, row in df['form'].items():
    df.iloc[i, form_index] = str.lower(row)

form_index = list(df.columns).index('lemma')
for i, row in df['lemma'].items():
    df.iloc[i, form_index] = str.lower(row)

upos = df.upos.unique()
# upos = upos.values
for i in upos:
    if i == 'VERB':
        continue
    else:
        df = df.drop(df[df.upos == i].index)
# df = df.drop(df[df.upos == "SYM"].index)
# df = df.drop(df[df.upos == "NUM"].index)

df.to_csv(
    path / "english_train.csv",
    index=False,
)
