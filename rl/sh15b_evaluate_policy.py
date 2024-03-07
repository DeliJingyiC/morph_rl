import sys
import pandas as pd
from collections import *
from pathlib import Path
from arguments import *
import numpy as np

def policyResult(block, policy="predicted"):
    for step, (ind, row) in enumerate(block.iterrows()):
        if policy == "predicted":
            act = row["predicted_optimal_action"]
        elif policy == "optimal":
            act = row["optimal_action"]
        elif policy == "stop":
            act = "stop"
        elif policy == "wait":
            act = "wait"
        else:
            assert(0)

        if step == (len(block) - 1):
            act = "stop" #end of the line

        if act == "stop":
            output = row["predicted"]
            target = row["form"]

            corr = (output == target)
            time = row["response_time"]

            return corr, time, step

    assert(0), "can't happen"

def describePolicy(data, policy="predicted", verbose=False):
    corrects = []
    times = []
    steps = []
    byFeats = defaultdict(list)

    for ind, ((inp, feats, trg), block) in enumerate(data.groupby(["lemma", "feats", "form"])):
        block = block.sort_values("response_time")
        correct, time, step = policyResult(block, policy=policy)
        if step > 0 and verbose is True:
            print("\t", "waited till step", step, "for", inp, feats, trg)
        if verbose == "features":
            byFeats[feats].append(step)
        corrects.append(correct)
        times.append(time)
        steps.append(step)

    corrects = np.array(corrects)
    times = np.array(times)
    steps = np.array(steps)

    print("results of policy", policy)
    print(np.sum(corrects), "/", len(corrects), np.mean(corrects), "items correct")
    print(np.mean(times), "time taken")
    print(np.sum((steps > 0)), "steps > 0")
    print(np.mean(steps), "avg step")

    if verbose == "features":
        def rate1(item):
            key, lst = item
            nPlus = len([xx for xx in lst if xx > 0])
            return nPlus / len(lst)

        for ft, ts in sorted(byFeats.items(), key=rate1, reverse=True):
            if len(ts) <= 1:
                continue

            ftS = ";".join(sorted(ft))

            nPlus = len([xx for xx in ts if xx > 0])
            m1 = np.mean(ts)
            m2 = np.median(ts)
            print(f"{ftS}\t#: {len(ts)}\t#>0: {nPlus} ({nPlus / len(ts)})\tavg: {m1}\tmed: {m2}")
        print()

    print()

if __name__ == '__main__':
    args = parseArgs()
    dataDir = Path(args.project) / "rl/dataset/"

    outputDir = Path(args.transformer_output) / "transformerregressor/sigmorphon17-task1-dropout0.3"

    language = args.language

    dataPath = dataDir / f"rewards_{language}_test.csv"
    data = pd.read_csv(dataPath)
    data.feats = data.feats.map(lambda xx: frozenset(eval(xx)))
    data.source_feats = data.source_feats.map(eval)

    run = Path(args.transformer_output).name.strip("ud_")    

    partition_out = outputDir / f"ud_{language}_reward-.decode.dev.tsv"
    with open(partition_out) as pfh:
        decode = pfh.readlines()
        decode = [line.strip("\n").split("\t") for line in decode]
        decode.pop(0) #remove tsv header "prediction", "target", "loss", "dist"
        preds = [pred.split(";") for (pred, trg, loss, dist) in decode]
        preds = np.array(preds, "float")

    data["predicted_reward_wait"] = preds[:, 0]
    data["predicted_reward_stop"] = preds[:, 1]

    data["predicted_optimal_action"] = None
    data.loc[data["predicted_reward_wait"] <= data["predicted_reward_stop"], "predicted_optimal_action"] = "wait"
    data.loc[data["predicted_reward_wait"] > data["predicted_reward_stop"], "predicted_optimal_action"] = "stop"

    data.to_csv(dataDir / f"{language}_policy_output.csv", index=False)

    verbose = "features"

    describePolicy(data, "predicted", verbose=verbose)
    describePolicy(data, "optimal", verbose=verbose)
    describePolicy(data, "stop", verbose=verbose)
    describePolicy(data, "wait", verbose=verbose)
