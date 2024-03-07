import sys
import pandas as pd
import numpy as np
from collections import *
from pathlib import Path
from arguments import *
from sh11b_encode_instances import *

def computeRewards(block, worst):
    block.sort_values("response_time", inplace=True)

    correct = (block["predicted"] == block["form"])

    # print(list(block.columns))
    #print(correct)

    block.loc[~correct, "reward_stop"] = worst
    #negative retrieval times don't make any sense
    block.loc[correct, "reward_stop"] = np.minimum(0, -block.loc[correct, "response_time"])

    # print(block[["predicted", "form"]])

    maxvals = block["reward_stop"][::-1].cummax()[::-1]
    maxvals = maxvals[1:].tolist() + [worst,]

    # print("cumulative max rewards", maxvals)

    block["reward_wait"] = maxvals
    block.loc[block["reward_wait"] > block["reward_stop"], "optimal_action"] = "wait"
    block.loc[block["reward_wait"] <= block["reward_stop"], "optimal_action"] = "stop"

    # if (block["optimal_action"] == "wait").any():
    #     print(block[["form", "predicted", "reward_wait", "reward_stop"]])
    #     input()

    return block

if __name__ == '__main__':
    args = parseArgs()
    dataDir = Path(args.project) / "rl/dataset/"
    language = args.language

    for split in ["dev", "test"]:
        dataPath = dataDir / f"prediction_output_{language}_{split}.csv"
        data = pd.read_csv(dataPath)
        worst = -data["response_time"].max()
        best = -data.loc[data["response_time"] > 0, "response_time"].min()

        print("worst time in the dataset:", worst)
        print("best time in the dataset:", best)

        data["reward_stop"] = None
        data["reward_wait"] = None
        data["optimal_action"] = None

        data = data.groupby(["lemma", "feats", "form"]).apply(lambda x: computeRewards(x, worst))

        #standardize rewards
        data.loc[data["reward_wait"] >= 0, "reward_wait"] = best
        # print("taking log of wait rewards")
        data["reward_wait"] = np.log((-data["reward_wait"].astype("float64")))
        data.loc[data["reward_stop"] >= 0, "reward_stop"] = best
        # print("taking log of stop rewards")
        # print(len(data.loc[data["reward_stop"] == 0, "reward_stop"]), "zeros")
        # print(len(data.loc[data["reward_stop"] == None, "reward_stop"]), "Nones")
        data["reward_stop"] = np.log((-data["reward_stop"].astype("float64")))
        rew = pd.concat([data["reward_wait"], data["reward_stop"]])
        mu = np.mean(rew)
        sig = np.std(rew)

        print("mean", mu, "std", sig)

        data["reward_wait"] = (data["reward_wait"] - mu) / sig
        data["reward_stop"] = (data["reward_stop"] - mu) / sig

        data.to_csv(dataDir / f"rewards_{language}_{split}.csv", index=False)
