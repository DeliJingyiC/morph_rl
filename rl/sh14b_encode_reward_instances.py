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

    for split in ["dev", "test"]:
        dataPath = dataDir / ("rewards_%s.csv" % split)

        if split == "dev":
            split = "train"

        data = pd.read_csv(dataPath)
        data.feats = data.feats.map(lambda xx: frozenset(eval(xx)))
        data.source_feats = data.source_feats.map(eval)

        instances = convert(data, single=args.single_source, cumulative=cumulative, cutoff=600, targetReward=True)

        sg = "reward"
            
        outf = neuralDir / (f"ud_{sg}-{split}")
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

                    charset = "".join(sorted(list(charset)))

                    for block in np.arange(0, len(charset), mx):
                        ofh.write("\t".join(["P!" + charset[block:block+mx], "0;0", "0"]) + "\n")

            if not args.multitask_only:
                for inst in instances:
                    ofh.write("\t".join(inst) + "\n")
