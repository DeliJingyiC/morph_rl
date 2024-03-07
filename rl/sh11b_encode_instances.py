import sys
import pandas as pd
from collections import *
from pathlib import Path
from arguments import *
import numpy as np

def processFeats(feats, lemma, featsep):
    # return featsep.join([xx for xx in feats])
    return featsep.join([xx for xx in sorted(feats) if xx != lemma])

def writeRow(lemma, feats, sources,
             featsep=";", lsep=".", infsep=">", instsep=":"):
    sourceStrings = [
        f"{srcLemma}{lsep}{processFeats(srcFeats, srcLemma, featsep)}{infsep}{srcForm}" for
        (srcLemma, srcFeats, srcForm)
        in sources if srcFeats is not False]
    fullStr = f"{lemma}{lsep}{processFeats(feats, lemma, featsep)}{instsep}{instsep.join(sourceStrings)}"
    return fullStr

def mapFeatsToChars(feats, feat_to_char, startCP=0x2100):
    res = []
    for feat in feats:
        if feat not in feat_to_char:
            index = startCP + len(feat_to_char)
            feat_to_char[feat] = chr(index)

        res.append(feat_to_char[feat])

    return res

def convert(data, single=False, cumulative=True, cutoff=600, targetReward=False, feat_to_char=None):
    instances = []
    data["pruned"] = False
    blocks = 0
    for (inp, feats, trg), block in data.groupby(["lemma", "feats", "form"]):
        blocks += 1
        sources = []

        if feat_to_char != None:
            feats = mapFeatsToChars(feats, feat_to_char)

        for index, source in block.sort_values("response_time").iterrows():
            if not cumulative:
                sources = []

            sourceFeats = source["source_feats"]
            if feat_to_char != None and sourceFeats != False:
                sourceFeats = mapFeatsToChars(sourceFeats, feat_to_char)

            sources.append((source["source_lemma"], sourceFeats, source["source_form"]))
            if (len(sources) == 1 and sources[0][1] is False) or not single:
                inst = writeRow(inp, feats, sources)

                if targetReward:
                    trg = source["reward_wait"], source["reward_stop"]
                    trg = ";".join([str(xx) for xx in trg])

                instances.append((inst, trg, "0"))

                if len(inst) >= cutoff:
                    #print("setting pruned", source)
                    data.loc[index, "pruned"] = True

    #apply length cutoff
    instances = [xx for xx in instances if len(xx[0]) < cutoff]

    print("read", blocks, "blocks and produced", len(instances), "instances")

    return instances

def randomWord(low, high, alphabet="abcedghijklmnopqrstuvwxyz", upper=False):
    length = np.random.choice(np.arange(low, high + 1))
    chrs = ""
    for li in range(length):
        chrs += np.random.choice(list(alphabet))

    if upper:
        chrs = chrs.upper()

    return chrs

def synthesize(nInst, copy=False, alphabet=None, feat_to_char=None):
    allFeats = [randomWord(1, 3, upper=True, alphabet=alphabet) for xx in range(10)]
    if feat_to_char != None:
        allFeats = list(feat_to_char.values())

    instances = []

    for ii in range(nInst):
        lemma = randomWord(4, 7, alphabet=alphabet)
        if copy:
            #create a totally suppletive instance
            #give the answer using the exemplar
            source = lemma
            target = randomWord(4, 7, alphabet=alphabet)
            sourceForm = target
        else:
            source = randomWord(4, 7, alphabet=alphabet)
            affix = randomWord(1, 3, alphabet=alphabet)

            if np.random.choice([0, 1]):
                sourceForm = source + affix
                target = lemma + affix
            else:
                sourceForm = affix + source
                target = affix + lemma

        gNum = np.random.choice(np.arange(1, 4))
        gFeats = np.random.choice(allFeats, size=gNum, replace=False)
        inst = writeRow(lemma, gFeats, [(source, gFeats, sourceForm)])
        instances.append((inst, target, "0"))

    return instances

def writeFeatToChar(of, feat_to_char):
    dicts = []
    for fi, ci in feat_to_char.items():
        pt = { "feature" : fi, "char" : ci, "charCode" : ord(ci) }
        dicts.append(pt)

    df = pd.DataFrame(dicts)
    df.to_csv(of, index=False)

if __name__ == '__main__':
    args = parseArgs()
    dataDir = Path(args.project) / "rl/dataset/"
    neuralDir = Path(args.project) / "neural-transducer/data/reinf_inst"
    cumulative = (not args.noncumulative and not args.single_source)
    language = args.language

    if args.char_encode_features:
        feat_to_char = {}
        fstr = "_char"
    else:
        feat_to_char = None
        fstr = ""

    for split in ["train", "dev", "test"]:
        dataPath = dataDir / (f"query_{language}_{split}.csv")
        data = pd.read_csv(dataPath)
        data.feats = data.feats.map(lambda xx: frozenset(eval(xx)))
        data.source_feats = data.source_feats.map(eval)

        instances = convert(data, single=args.single_source, cumulative=cumulative, cutoff=600,
                            feat_to_char=feat_to_char)

        if args.single_source:
            sg = "single"
        elif args.multitask_only:
            sg = "synthetic"
        else:
            sg = "multi"
            
        outf = neuralDir / (f"ud_{language}_{sg}{fstr}-{split}")
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
                        ofh.write("\t".join(["P!" + "".join(charset[block:block+mx]), "".join(charset[block:block+mx]), "0"]) + "\n")

                if split == "train" or ((split == "dev" or split == "test") and args.multitask_only):
                    if feat_to_char != None:
                        alphabet = set([xx for xx in charset if xx not in set(feat_to_char.values())])
                    else:
                        alphabet = set(charset)
                    alphabet.difference_update([";", ".", ">", ":", "!"])

                    alphabet = "".join(alphabet)

                    if args.multitask_only:
                        mult = 1 #10
                    else:
                        mult = 1
                    synthData = synthesize(mult * len(instances), alphabet=alphabet,
                                           feat_to_char=feat_to_char)
                    synthData = [("P!" + xx[0], xx[1], xx[2]) for xx in synthData]
                    for inst in synthData:
                        ofh.write("\t".join(inst) + "\n")

                    # synthData = synthesize(len(instances), copy=True)
                    # synthData = [("P!" + xx[0], xx[1], xx[2]) for xx in synthData]
                    # for inst in synthData:
                    #     ofh.write("\t".join(inst) + "\n")

            if not args.multitask_only:
                for inst in instances:
                    ofh.write("\t".join(inst) + "\n")

    if feat_to_char != None:
        of = neuralDir / (f"ud_{language}_{sg}{fstr}-mapping.csv")
        writeFeatToChar(of, feat_to_char)
