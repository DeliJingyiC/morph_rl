import sys
import pandas as pd
from collections import *
from pathlib import Path
from arguments import *
from sh11b_encode_instances import *

def computeRewards(block, worst):
    block.sort_values("response_time", inplace=True)

    correct = (block["predicted"] == block["form"])

    #print(correct)

    block.loc[~correct, "reward_stop"] = worst
    block.loc[correct, "reward_stop"] = -block.loc[correct, "response_time"]

    #print(block)

    maxvals = block["reward_stop"][::-1].cummax()[::-1]
    maxvals = maxvals[1:].tolist() + [worst,]

    #print("cumulative max rewards", maxvals)

    block["reward_wait"] = maxvals
    block.loc[block["reward_wait"] > block["reward_stop"], "optimal_action"] = "wait"
    block.loc[block["reward_wait"] <= block["reward_stop"], "optimal_action"] = "stop"

    return block

if __name__ == '__main__':
    args = parseArgs()
    dataDir = Path(args.project) / "rl/dataset/"

    for split in ["dev", "test"]:
        dataPath = dataDir / ("prediction_output_%s.csv" % split)
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
        data.loc[data["reward_wait"] == 0, "reward_wait"] = best
        data["reward_wait"] = np.log((-data["reward_wait"].astype("float64")))
        data.loc[data["reward_stop"] == 0, "reward_stop"] = best
        data["reward_stop"] = np.log((-data["reward_stop"].astype("float64")))
        rew = pd.concat([data["reward_wait"], data["reward_stop"]])
        mu = np.mean(rew)
        sig = np.std(rew)
        data["reward_wait"] = (data["reward_wait"] - mu) / sig
        data["reward_stop"] = (data["reward_stop"] - mu) / sig

        data.to_csv(dataDir / ("rewards_%s.csv" % split), index=False)
