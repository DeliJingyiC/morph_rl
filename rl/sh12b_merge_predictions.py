import sys
import pandas as pd
from collections import *
from pathlib import Path
from arguments import *
from sh11b_encode_instances import *

if __name__ == '__main__':
    args = parseArgs()
    dataDir = Path(args.project) / "rl/dataset/"
    neuralDir = Path(args.project) / "neural-transducer/data/reinf_inst"
    outputDir = Path(args.transformer_output) / "tagtransformer/sigmorphon17-task1-dropout0.3"

    run = Path(args.transformer_output).name.strip("ud_")    
    language = run.replace("_multi", "").replace("_char", "")

    for split in ["dev", "test"]:
        dataPath = dataDir / (f"query_{language}_{split}.csv")
        data = pd.read_csv(dataPath)
        data.feats = data.feats.map(lambda xx: frozenset(eval(xx)))
        data.source_feats = data.source_feats.map(eval)
        data["predicted"] = None

        #XXX; did some runs with the filenames swapped to reduce decode times
        if split == "dev":
            split2 = "test"
        else:
            split2 = "dev"

        instances = convert(data, single=args.single_source, cutoff=600, cumulative=False) #set pruned attribute
        partition_out = outputDir / (f"ud_{run}-.decode.{split2}.tsv")
        with open(partition_out) as pfh:
            decode = pfh.readlines()
            decode = [line.strip("\n").split("\t") for line in decode]
            decode.pop(0) #remove tsv header "prediction", "target", "loss", "dist"
            decode = ["".join(pred.split()) for (pred, trg, loss, dist) in decode]

            #fix later--- dump lines used to load charset
            # decode.pop(0)
            # decode.pop(0)

        # if split == "dev": #dump synthetic examples
        #     unpruned = len(data.loc[~data.pruned])

        #     print("current dev lines", len(decode))
        #     print("current data:", len(data), "unpruned", unpruned)

        #     decode = decode[-unpruned:]

        #     print("reduced length of dev decode to", unpruned)
        #     print("first 5 of decode dev")
        #     print(decode[:5])

        ctr = 0
        last = 0
        for (inp, feats, trg), block in data.groupby(["lemma", "feats", "form"]):
            bSort = block.sort_values("response_time")
            bSort = bSort.loc[~bSort.pruned]
            index = bSort.index
            decodes = decode[ctr : ctr + len(index)]
            #print("ind", len(index), "dec", len(decodes), "ct", ctr)
            data.loc[index, "predicted"] = decodes
            ctr += len(decodes)

            # print(data.loc[index, ["lemma", "form", "predicted"]])
            # assert(0)

            if (ctr - last) > 1000:
                print(ctr, ":", len(decode))
                last = ctr


            # for index, row in block.sort_values("response_time").iterrows():
            #     if not row["pruned"]:
            #         data.loc[index, "predicted"] = decode[ctr]
            #         ctr += 1

            #         if ctr % 1000 == 0:
            #             print(ctr, len(decode))

        # for ind, ((inp, feats, trg), block) in enumerate(data.groupby(["lemma", "feats", "form"])):
        #     print(block)
        #     if ind > 5:
        #         break
        print("in split", split, len(decode), "decoded lines but", ctr, "were used to match", len(data), "elements of which", len(data[~data.pruned]), "were not pruned")
        assert(ctr == len(decode))

        multipred = 0
        multiAndRight = 0
        for ind, ((inp, feats, trg), block) in enumerate(data.groupby(["lemma", "feats", "form"])):
            unpruned = block.loc[~block.pruned]
            #print(unpruned)
            preds = unpruned.predicted
            upreds = preds.unique()
            if len(upreds) > 1:
                if trg in upreds:
                    multiAndRight += 1
                # print("multiple predictions at")
                # print(block)
                multipred += 1

        print("For split", split, "multiple predictions per block in", multipred, "blocks of", len(data.groupby(["lemma", "feats", "form"])))
        print("In", multiAndRight, "we eventually get it")

        data.to_csv(dataDir / (f"prediction_output_{language}_{split}.csv"), index=False)
