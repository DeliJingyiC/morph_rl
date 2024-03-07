import sys
import pandas as pd
from collections import *
from pathlib import Path
from arguments import *
import numpy as np

from sh11b_encode_instances import convert

if __name__ == '__main__':
    args = parseArgs()
    dataDir = Path(args.project) / "rl/dataset/"
    neuralDir = Path(args.project) / "neural-transducer/data/reinf_inst"
    cumulative = not args.noncumulative
    language = args.language

    if args.char_encode_features:
        feat_to_char = {}
        featsFile = neuralDir / f"ud_{language}_multi_char-mapping.csv"
        fData = pd.read_csv(featsFile)

        for ind, row in fData.iterrows():
            feat_to_char[row["feature"]] = row["char"]
    else:
        feat_to_char = None

    for split in ["dev", "test"]:
        dataPath = dataDir / (f"rewards_{language}_{split}.csv")

        if split == "dev":
            split = "train"

        data = pd.read_csv(dataPath)
        data.feats = data.feats.map(lambda xx: frozenset(eval(xx)))
        data.source_feats = data.source_feats.map(eval)

        instances = convert(data, single=args.single_source, cumulative=cumulative, 
                            cutoff=600, targetReward=True, feat_to_char=feat_to_char)

        sg = "reward"
            
        outf = neuralDir / (f"ud_{language}_{sg}-{split}")
        with open(outf, "w") as ofh:
            if args.synthetic_multitask:
                instances = [("R!" + xx[0], xx[1], xx[2]) for xx in instances]
            
                if split == "train":
                    charset = set("abcedghijklmnopqrstuvwxyz" + "abcedghijklmnopqrstuvwxyz".upper() + "0")
                    mx = 0
                    for src, trg, feats in instances:
                        charset.update(src)
                        charset.update(trg)
                        charset.update(feats)
                        mx = max(mx, len(src))

                    charset = sorted("".join(list(charset)))

                    for block in np.arange(0, len(charset), mx):
                        ofh.write("\t".join(["P!" + "".join(charset[block:block+mx]), "0;0", "0"]) + "\n")

            if not args.multitask_only:
                for inst in instances:
                    ofh.write("\t".join(inst) + "\n")
