import sys
import os
import re
import math
import pandas as pd
from collections import *
from pathlib import Path
import warnings
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from arguments import *

from transformer import *
import util
from transformer_adaptors import *

from transducers import formatInputs, convertFromEdits

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Simulator:
    def __init__(self, dataHandler, model, train):
        self.data = dataHandler
        self.model = model
        self.worst = -train["response_time"].max()
        self.best = -train.loc[train["response_time"] > 0, "response_time"].min()

        self.mean = 0
        self.nn = 0
        self.msq = 0

        self.responseTimes = -train["response_time"].to_numpy()
        self.responseTimes[self.responseTimes > 0] = self.best
        self.responseTimes = np.sort(self.responseTimes)

        # print("first 50 response times", self.responseTimes[:50])
        # print("last 50 response times", self.responseTimes[-50:])

    def simulate(self, block):
        (lemma, form, feats), block = block
        block = block.reset_index()
        instances = self.data.blockToInstances(lemma, form, feats, block)

        # print("encoded instances")
        # for xx in instances:
        #    print(xx)
        # print()
        # assert(0)

        tensors = self.data.instancesToTensors(instances)
        rawPredictions = self.model.stringPredictions(tensors)
        # print("raw predictions", predictions)
        rawPredictions = self.spacePrunedValues(rawPredictions, instances)
        predictions = self.data.decipherPredictions(lemma, form, feats, block, rawPredictions)
        # print("deciphered", predictions)
        self.data.updateDynamicTargets(lemma, form, feats, block, rawPredictions, predictions)
        qpreds = self.model.valuePredictions(tensors)
        qpreds = pd.DataFrame(qpreds, columns=["reward_stop", "reward_wait"])
        rewards = self.calcRewards(block, instances, predictions)
        rewards["pred_reward_stop"] = qpreds["reward_stop"]
        rewards["pred_reward_wait"] = qpreds["reward_wait"]

        rewards.loc[rewards["pred_reward_wait"] > rewards["pred_reward_stop"], "predicted_action"] = "wait"
        rewards.loc[rewards["pred_reward_wait"] <= rewards["pred_reward_stop"], "predicted_action"] = "stop"

        # if (rewards["predicted_action"] != rewards["optimal_action"]).all():
        #     rewards["raw"] = rawPredictions
        #     print(lemma, form, feats)
        #     print(rewards)
        #     normR = self.normalizeReward(rewards, verbose=True)
        #     print("normed reward")
        #     print(normR)

        stateProbs = self.rewardsToProbs(qpreds)

        rewards["pr_state"] = stateProbs
        rewards["instance"] = instances

        wait, time, correct = self.evaluatePolicy(rewards, policy="predicted")

        return rewards, (wait, correct)

    def calcRewards(self, block, instances, predictions):
        correct = (predictions == block["form"])

        # print("correct?", correct)

        rewardDF = pd.DataFrame({ "correct" : correct, "prediction" : predictions})

        rewardDF["reward_stop"] = [None for xx in correct]
        rewardDF.loc[~correct, "reward_stop"] = self.worst
        rewardDF.loc[correct, "reward_stop"] = -block.loc[correct, "response_time"]

        # print(block["response_time"])
        # print(rewardDF) ##.loc[:, ["prediction", "correct", "reward_stop"]])

        maxvals = rewardDF["reward_stop"][::-1].cummax()[::-1]
        maxvals = maxvals[1:].tolist() + [self.worst,]

        # print("cumulative max rewards", maxvals)

        rewardDF["reward_wait"] = maxvals
        rewardDF.loc[rewardDF["reward_wait"] > rewardDF["reward_stop"], "optimal_action"] = "wait"
        rewardDF.loc[rewardDF["reward_wait"] <= rewardDF["reward_stop"], "optimal_action"] = "stop"

        return rewardDF

    def spacePrunedValues(self, predictions, instances):
        pruned = [xx[-1] for xx in instances]
        pred = iter(predictions)

        # print("spacing values:", pruned, len(predictions), len(pruned))

        augmented = [next(pred) if not prune else None for prune in pruned]
        return augmented

    def rewardsToProbs(self, rewards):
        norms = rewards["reward_wait"] + rewards["reward_stop"]
        rewards["pr_stop"] = np.clip(rewards["reward_stop"] / norms, .05, .95)
        rewards.loc[len(rewards) - 1, "pr_stop"] = 1
        prReachState = pd.concat([pd.Series(1), (1 - rewards["pr_stop"]).cumprod()],
                                 ignore_index=True)

        rewards["pr_reach_state"] = prReachState

        prEndState = prReachState[:-1] * rewards["pr_stop"]

        # print(rewards)

        return prEndState

    def normalizeReward(self, data, verbose=False):
        return self.normalizeRewardCopula(data, verbose)

    def normalizeRewardWhiten(self, data, verbose=False):
        #make sure all rewards are negative and lie between worst and best
        #update sufficient stats to whiten
        #apply whitening transform
        rw = data["reward_wait"].to_numpy(dtype="float32")
        rw[rw >= 0] = self.best
        rs = data["reward_stop"].to_numpy(dtype="float32")
        rs[rs >= 0] = self.best

        allR = np.concatenate([rw, rs])
        for xi in allR:
            self.updateRewardMoments(xi)

        mu, sigma = self.rewardMoments()

        if verbose == True:
            print("rw", rw)
            print("rs", rs)
            print(self.best)
            print("moments reported", mu, sigma)

        rw = (rw - mu) / sigma
        rs = (rs - mu) / sigma

        # print("stacking", rw.shape, rs.shape)
        res = np.stack([rs, rw], axis=-1)
        # print(res.shape, res.dtype)

        return res

    def normalizeRewardCopula(self, data, verbose=False):
        #make sure all rewards are negative and lie between worst and best
        #update sufficient stats to whiten
        #apply whitening transform
        rw = data["reward_wait"].to_numpy(dtype="float32")
        rw[rw > 0] = self.best
        rs = data["reward_stop"].to_numpy(dtype="float32")
        rs[rs > 0] = self.best

        indW = np.searchsorted(self.responseTimes, rw)
        indS = np.searchsorted(self.responseTimes, rs)
        nn = self.responseTimes.shape[0]
        propW = (indW / nn).astype("float32")
        propS = (indS / nn).astype("float32")

        #order must match dataframe construction around l65
        #which must also match model target invocation in learnValues
        #would be better to return this as a named structure?
        res = np.stack([propS, propW], axis=-1)

        if verbose == True:
            print("rw", rw)
            print("rs", rs)
            print(self.best)
            print("index in sorted list of w", indW)
            print("converted to 0-1", propW)
            print("index in sorted list of s", indS)
            print("converted to 0-1", propS)

        return res        

    def actionVector(self, data, verbose=False):
        rW = data["reward_wait"].to_numpy(dtype="float32")
        rS = data["reward_stop"].to_numpy(dtype="float32")

        optS = (rS > rW).astype("float32")
        optS[rS == rW] = .5
        optW = (1 - optS)

        #order must match dataframe construction around l65
        #which must also match model target invocation in learnValues
        #would be better to return this as a named structure?
        res = np.stack([optS, optW], axis=-1)

        return res        

    def evaluatePolicy(self, rewards, policy):
        if policy == "predicted":
            stop = "pred_reward_stop"
            wait = "pred_reward_wait"

            crit = lambda ri: ri[stop] > ri[wait]
        elif policy == "optimal":
            stop = "reward_stop"
            wait = "reward_wait"

            crit = lambda ri: ri[stop] > ri[wait]
        elif policy == "stop":
            crit = lambda ri: True
        elif policy == "wait":
            crit = lambda ri: False
        else:
            assert(0), "unknown policy type"

        for step, ri in rewards.iterrows():
            if crit(ri):
                return step, ri["reward_stop"], ri["correct"]

        return step, ri["reward_stop"], ri["correct"]

    def updateRewardMoments(self, sample):
        # rns = np.random.uniform(size=10)
        # for ii in rns:
        #     self.dataHandler.updateRewardMoments(ii)
        # mean, var = self.dataHandler.rewardMoments()
        # print(mean, var)
        # print(np.mean(rns), np.var(rns, ddof=1))
        # assert(0)
        # https://stats.stackexchange.com/questions/235129/online-estimation-of-variance-with-limited-memory

        self.nn += 1
        if self.nn == 1:
            self.mean = sample
        else:
            delta = (sample - self.mean)
            self.mean += delta / self.nn
            self.msq += delta * (sample - self.mean)

        #print("reward moments:", self.nn, self.mean, self.msq)

    def rewardMoments(self):
        if self.nn > 1:
            var = self.msq / (self.nn - 1)
            var = np.maximum(.1, var)
            return self.mean, var
        else:
            return self.mean, 1

class DataHandler:
    def __init__(self, mode="create", train=None, cumulative=False, cutoff=600, settings=None):
        self.startCP=0x2100
        self.cumulative = cumulative
        self.cutoff = cutoff

        if mode == "create":
            self.readCharset(train)
        else:
            self.maxLenSrc = settings["maxLenSrc"]
            self.maxLenTarg = settings["maxLenTarg"]
            self.sourceC2I = settings["sourceVocab"]
            self.targC2I = settings["targVocab"]
            self.featToChar = settings["featToChar"]

    def readCharset(self, train):
        self.maxLenSrc = 0
        self.maxLenTarg = 0

        self.sourceC2I = {
            "<PAD>" : 0,
            "<BOS>" : 1,
            "<EOS>" : 2,
            "<UNK>" : 3,
            ";" : 4,
            "." : 5,
            ">" : 6,
            ":" : 7
            }

        self.targC2I = {
            "<PAD>" : 0,
            "<BOS>" : 1,
            "<EOS>" : 2,
            "<UNK>" : 3,
        }

        self.featToChar = { }

        for ind, row in train.iterrows():
            src = row["lemma"] + row["source_lemma"] + row["source_form"]
            ln = len(src) + 2 * len(row["feats"])
            if row["source_feats"] != False:
                ln += 2 * len(row["source_feats"])
            ln += 4
            self.maxLenSrc = max(ln, self.maxLenSrc)

            for ci in src:
                if ci not in self.sourceC2I:
                    self.sourceC2I[ci] = len(self.sourceC2I)

            trg = row["form"]
            #bos/eos decorators
            self.maxLenTarg = max(4 + 2 * len(trg), self.maxLenTarg)
            for ci in trg:
                if ci not in self.targC2I:
                    self.targC2I[ci] = len(self.targC2I)

            feats = row["feats"]
            if row["source_feats"] != False:
                feats = feats.union(row["source_feats"])
            if feats != None:
                for fi in feats:
                    if fi not in self.featToChar:
                        index = self.startCP + len(self.featToChar)
                        self.featToChar[fi] = chr(index)
                        self.sourceC2I[chr(index)] = len(self.sourceC2I)

        print("DataHandler character sets:")
        print(self.sourceC2I)
        print(self.targC2I)
        print(self.featToChar)

    def sourceVocabSize(self):
        return 1 + len(self.sourceC2I)

    def targVocabSize(self):
        return 1 + len(self.targC2I)

    def padAndEncode(self, strings, c2i):
        res = []
        unk = c2i["<UNK>"]
        for si in strings:
            si = ["<BOS>"] + list(si) + ["<EOS>"]
            inds = [c2i.get(ci, unk) for ci in si]
            res.append(inds)

        return res

    def stripEOS(self, seq):
        try:
            eos = seq.index("<EOS>")
        except ValueError:
            eos = len(seq)

        trimmed = seq[:eos]
        return [ch for ch in trimmed if ch not in ["<BOS>", "<PAD>"]]

    def decode(self, inds, c2i):
        i2c = dict({ vv : kk for (kk, vv) in c2i.items() })
        #print("i2c", i2c)
        chs = [[i2c.get(ind, "?") for ind in seq] for seq in inds]
        #print(chs)

        chs = [self.stripEOS(ch) for ch in chs]
        return ["".join(ch) for ch in chs]

    def defeaturize(self, chrs):
        f2name = dict({ vv : kk for (kk, vv) in self.featToChar.items() })
        chrs = [f2name.get(ch, ch) for ch in chrs]
        return "".join(chrs)

    def listToTensor(self, inds, maxLen):
        data = torch.zeros((maxLen, len(inds)), dtype=torch.long).to(DEVICE)
        # print("DEVICE is", DEVICE)
        # for dev in range(torch.cuda.device_count()):
        #     print(torch.cuda.get_device_name(dev))
        # print("device check for new tensor", data.get_device())
        for ii, seq in enumerate(inds):
            data[: len(seq), ii] = torch.tensor(seq)
        mask = (data > 0).float()
        # print("device check for mask", mask.get_device())
        return data, mask

    def instancesToTensors(self, instances):
        # print("CONVERTING", len(instances), "to tensors with max length", self.maxLenSrc, self.maxLenTarg)

        sources = [src for src, trg, _, prune in instances if not prune]
        targets = [trg for src, trg, _, prune in instances if not prune]

        srcInds = self.padAndEncode(sources, self.sourceC2I)
        targInds = self.padAndEncode(targets, self.targC2I)

        #debug block in case strlengths were calculated wrongly
        for si, ii in zip(srcInds, sources):
            if len(si) >= self.maxLenSrc:
                print("long seq:", len(si), len(ii), self.maxLenSrc)
                print(si, ii)
        for si, ii in zip(targInds, targets):
            if len(si) >= self.maxLenTarg:
                print("long seq:", len(si), self.maxLenTarg)
                print(si, ii)

        srcTensor, srcMask = self.listToTensor(srcInds, self.maxLenSrc)
        targTensor, targMask = self.listToTensor(targInds, self.maxLenTarg)

        return srcTensor, srcMask, targTensor, targMask

    def processFeats(self, feats, lemma, featsep):
        # return featsep.join([xx for xx in feats])
        return featsep.join([xx for xx in sorted(feats) if xx != lemma])

    def writeRow(self, lemma, feats, sources,
             featsep=";", lsep=".", infsep=">", instsep=":"):
        sourceStrings = [
            f"{srcLemma}{lsep}{self.processFeats(srcFeats, srcLemma, featsep)}{infsep}{srcForm}" for
            (srcLemma, srcFeats, srcForm)
            in sources if srcFeats is not False]
        fullStr = f"{lemma}{lsep}{self.processFeats(feats, lemma, featsep)}{instsep}{instsep.join(sourceStrings)}"
        return fullStr

    def mapFeatsToChars(self, feats):
        res = []
        for feat in feats:
            if feat in self.featToChar:
                res.append(self.featToChar[feat])

        return tuple(res)

    def blockToInstances(self, lemma, trg, feats, block):
        if self.featToChar != None:
            feats = self.mapFeatsToChars(feats)

        sources = []
        instances = []
        for index, source in block.iterrows():
            if not self.cumulative:
                sources = []

            sourceFeats = source["source_feats"]
            if self.featToChar != None and sourceFeats != False:
                sourceFeats = self.mapFeatsToChars(sourceFeats)

            sources.append((source["source_lemma"], sourceFeats, source["source_form"]))
            inst = self.writeRow(lemma, feats, sources)
            instances.append((inst, trg, "0", len(inst) >= self.cutoff))

        return instances

    def rewardToTensors(self, rewards):
        # print("convert to tensors")
        # print(rewards)
        targets = torch.from_numpy(rewards).transpose(0, 1).to(DEVICE)
        mask = torch.ones((1, rewards.shape[0]), dtype=torch.float)
        return targets, mask        

    def decipherPredictions(self, lemma, form, feats, block, predictions):
        return predictions

    def updateDynamicTargets(self, lemma, form, feats, block, rawPredictions, predictions):
        return

class EditHandler(DataHandler):
    def __init__(self, mode="create", train=None, cumulative=False, cutoff=600, settings=None):
        super(EditHandler, self).__init__(mode, train, cumulative, cutoff, settings)
        self.analyses = {}
    
    def readCharset(self, train):
        super().readCharset(train)
        for sym in ["0", "1", "-0", "-1"]:
            if sym not in self.targC2I:
                self.targC2I[sym] = len(self.targC2I)

    def blockToInstances(self, lemma, trg, feats, block):
        if self.featToChar != None:
            feats = self.mapFeatsToChars(feats)

        sources = []
        instances = []
        for index, source in block.iterrows():
            if not self.cumulative:
                sources = []

            sourceFeats = source["source_feats"]
            if self.featToChar != None and sourceFeats != False:
                sourceFeats = self.mapFeatsToChars(sourceFeats)

            sources.append((source["source_lemma"], sourceFeats, source["source_form"]))

            key = (lemma, trg, feats, index)
            if key in self.analyses:
                instance, diffedSources = self.analyses[key]
            else:
                instance, diffedSources = self.createInstance(lemma, trg, feats, sources)
                self.analyses[key] = (instance, diffedSources)

            instances.append(instance)

        return instances

    def padAndEncode(self, strings, c2i):
        res = []
        unk = c2i["<UNK>"]
        for si in strings:
            chrs = [xx for xx in re.split("([+-]?.)", si) if xx != ""]
            si = ["<BOS>"] + chrs + ["<EOS>"]
            inds = [c2i.get(ci, unk) for ci in si]
            res.append(inds)

            # print("indices for", si)
            # print("split into", chrs)
            # print("indices", inds)
            # print()

        return res

    def createInstance(self, lemma, trg, feats, sources):
        analysis, diffedExes = formatInputs(lemma, trg, 
                                            [(srcL, srcT) for (srcL, srcF, srcT) in sources])
        diffedSources = [(si, sf, diff) for ((si, sf, st), (diffSrc, diff)) in zip(sources, diffedExes)]
        inst = self.writeRow(lemma, feats, diffedSources)

        # print("instance:", inst)
        # print("targ:", analysis)
        # print()

        instance = ((inst, analysis, "0", len(inst) >= self.cutoff))
        return instance, diffedSources        

    def decipherPredictions(self, lemma, form, feats, block, predictions):
        res = []

        if self.featToChar != None:
            feats = self.mapFeatsToChars(feats)

        for (index, source), pred in zip(block.iterrows(), predictions):
            key = (lemma, form, feats, index)
            (instance, diffedSources) = self.analyses[key]
            if pred != None:
                stringResult = convertFromEdits(pred, lemma, diffedSources)
                (src, targ, _, _) = instance
                # print("tc2i", self.targC2I)
                # print("src:", src)
                # print("targ:", targ, "actual form:", form)
                # print("predicted:", pred)
                # print("converted from edits on:", lemma, diffedSources)
                # print("str:", stringResult)
                # print()
            else:
                stringResult = None

            res.append(stringResult)

        return res

    def updateDynamicTargets(self, lemma, form, feats, block, rawPredictions, predictions):
        return #current target updates are gross

        # if self.featToChar != None:
        #     feats = self.mapFeatsToChars(feats)

        # for (index, source), pred, raw in zip(block.iterrows(), predictions, rawPredictions):
        #     key = (lemma, form, feats, index)
        #     if pred == form:
        #         if len(raw) + 2 < self.maxLenTarg:
        #             (instance, diffedSources) = self.analyses[key]
        #             (inst, analysis, ftField, pruned) = instance
        #             instance = (inst, raw, ftField, pruned)
        #             self.analyses[key] = (instance, diffedSources)

class Model:
    def __init__(self, mode="create",
                 *,
                 src_vocab_size,
                 trg_vocab_size,
                 embed_dim=0,
                 nb_heads=None,
                 src_hid_size=None,
                 src_nb_layers=None,
                 trg_hid_size=None,
                 trg_nb_layers=None,
                 dropout_p=None,
                 tie_trg_embed=None,
                 label_smooth=None,
                 n_actions=None,
                 data,
                 max_norm=0,
                 warmup=4000,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.98,
                 value_mode="regress",
                 **kwargs
             ):
        super().__init__()
        self.data = data
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.nb_heads = nb_heads
        self.src_hid_size = src_hid_size
        self.src_nb_layers = src_nb_layers
        self.trg_hid_size = trg_hid_size
        self.trg_nb_layers = trg_nb_layers
        self.dropout_p = dropout_p
        self.tie_trg_embed = tie_trg_embed
        self.label_smooth = label_smooth
        self.n_actions = n_actions
        self.max_norm = max_norm
        self.value_mode = value_mode

        self.stringLosses = []
        self.valueLosses = []

        if mode == "create":
            self.src_embed = Embedding(src_vocab_size, embed_dim, padding_idx=PAD_IDX)
            self.trg_embed = Embedding(trg_vocab_size, embed_dim, padding_idx=PAD_IDX)
            self.position_embed = SinusoidalPositionalEmbedding(embed_dim, PAD_IDX)

            encoder_layer = TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nb_heads,
                dim_feedforward=src_hid_size,
                dropout=dropout_p,
                attention_dropout=dropout_p,
                activation_dropout=dropout_p,
                normalize_before=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=src_nb_layers, norm=nn.LayerNorm(embed_dim)
            )
            decoder_layer = TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=nb_heads,
                dim_feedforward=trg_hid_size,
                dropout=dropout_p,
                attention_dropout=dropout_p,
                activation_dropout=dropout_p,
                normalize_before=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=trg_nb_layers, norm=nn.LayerNorm(embed_dim)
            )
            self.final_out = Linear(embed_dim, trg_vocab_size)
            if tie_trg_embed:
                self.final_out.weight = self.trg_embed.weight
            self.dropout = nn.Dropout(dropout_p)
    
            r_decoder_layer = TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=nb_heads,
                dim_feedforward=trg_hid_size,
                dropout=dropout_p,
                attention_dropout=dropout_p,
                activation_dropout=dropout_p,
                normalize_before=True,
            )
            self.regressor_decoder = nn.TransformerDecoder(
                r_decoder_layer, num_layers=trg_nb_layers, norm=nn.LayerNorm(embed_dim)
            )
            self.r_final_out = Linear(embed_dim, self.n_actions)
    
            self.stringTransformer = TransformerFromLayers(encoder=self.encoder,
                                                           src_embed=self.src_embed,
                                                           decoder=self.decoder,
                                                           trg_embed=self.trg_embed,
                                                           position_embed=self.position_embed,
                                                           final_out=self.final_out,
                                                           dropout=self.dropout,
                                                           embed_scale=self.embed_scale,
                                                           trg_vocab_size=trg_vocab_size)
            self.stringTransformer.to(DEVICE)
    
            self.stringOptimizer = torch.optim.Adam(
                    self.stringTransformer.parameters(), lr, betas=(beta1, beta2)
                )
    
            self.stringScheduler = util.WarmupInverseSquareRootSchedule(
                    self.stringOptimizer, warmup
                )

            if self.value_mode == "regress":
                self.valueTransformer = TransformerRegressorFromLayers(encoder=self.encoder,
                                                                       src_embed=self.src_embed,
                                                                       decoder=self.regressor_decoder,
                                                                       trg_embed=self.trg_embed,
                                                                       position_embed=self.position_embed,
                                                                       final_out=self.r_final_out,
                                                                       dropout=self.dropout,
                                                                       embed_scale=self.embed_scale)
            elif self.value_mode == "classify":
                self.valueTransformer = TransformerClassifierFromLayers(encoder=self.encoder,
                                                                       src_embed=self.src_embed,
                                                                       decoder=self.regressor_decoder,
                                                                       trg_embed=self.trg_embed,
                                                                       position_embed=self.position_embed,
                                                                       final_out=self.r_final_out,
                                                                       dropout=self.dropout,
                                                                       embed_scale=self.embed_scale)


            self.valueTransformer.to(DEVICE)
    
            self.valueOptimizer = torch.optim.Adam(
                    self.valueTransformer.parameters(), lr, betas=(beta1, beta2)
                )
    
            self.valueScheduler = util.WarmupInverseSquareRootSchedule(
                    self.valueOptimizer, warmup
                )
        else:
            assert(mode == "load" and kwargs["load_model"] != None)
            loadf = kwargs["load_model"]
            self.stringTransformer = torch.load(f"{loadf}.params.string", map_location=DEVICE)
            self.valueTransformer = torch.load(f"{loadf}.params.value", map_location=DEVICE)

            self.valueTransformer.encoder = self.stringTransformer.encoder
            self.valueTransformer.src_embed = self.stringTransformer.src_embed

            self.stringTransformer.to(DEVICE)
    
            self.stringOptimizer = torch.optim.Adam(
                    self.stringTransformer.parameters(), lr, betas=(beta1, beta2)
                )
    
            self.stringScheduler = util.WarmupInverseSquareRootSchedule(
                    self.stringOptimizer, warmup
                )

            self.valueTransformer.to(DEVICE)

            self.valueOptimizer = torch.optim.Adam(
                    self.valueTransformer.parameters(), lr, betas=(beta1, beta2)
                )
    
            self.valueScheduler = util.WarmupInverseSquareRootSchedule(
                    self.valueOptimizer, warmup
                )

    def stringPredictions(self, tensors):
        # for ind, tns in enumerate(tensors):
        #     print("device check", ind, tns.get_device())

        preds = self.stringTransformer.forward(*tensors)

        #seq length x instances x charset
        #print("shape of preds", preds.shape)
        preds = preds.transpose(0, 1)
        ams = np.argmax(preds.cpu().detach().numpy(), axis=-1)
        decs = self.data.decode(ams, self.data.targC2I)
        return decs

    def valuePredictions(self, tensors):
        (src, srcMask, _, _) = tensors
        trg = torch.ones((1, src.shape[1]), dtype=torch.long).to(DEVICE)
        trgMask = torch.ones((1, src.shape[1])).to(DEVICE)

        preds = self.valueTransformer.forward(src, srcMask, trg, trgMask)
        #squeeze sequence dimension yielding batch x actions
        preds = preds.squeeze().cpu().detach().numpy()
        #print("preds", preds.shape)
        #print(preds)
        return preds

    def trainStringBatch(self, tensors):
        # for ii, ti in enumerate(tensors):
        #     print("shape of tensor", ii, "\t", ti.shape)

        loss = self.stringTransformer.get_loss(tensors)
        self.stringOptimizer.zero_grad()
        loss.backward()

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.stringTransformer.parameters(), max_norm)

        self.stringOptimizer.step()
        if not isinstance(self.stringScheduler, ReduceLROnPlateau):
            self.stringScheduler.step()
        self.stringLosses.append(loss.item())

        if len(self.stringLosses) > 1000:
            self.stringLosses.pop(0)

    def trainValueBatch(self, tensors):
        (src, srcMask, actualTrg, actualTrgMask) = tensors

        trg = torch.ones((1, src.shape[1]), dtype=torch.long).to(DEVICE)
        trgMask = torch.ones((1, src.shape[1])).to(DEVICE)
        tensors = (src, srcMask, trg, trgMask)

        # for ii, ti in enumerate(tensors):
        #     print("shape of tensor", ii, "\t", ti.shape)

        out = self.valueTransformer.forward(src, srcMask, trg, trgMask)
        # print("shape of predictions", out.shape)
        loss = self.valueTransformer.loss(out, actualTrg, reduction=True)
        self.valueOptimizer.zero_grad()
        loss.backward()

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.valueTransformer.parameters(), max_norm)

        self.valueOptimizer.step()
        if not isinstance(self.valueScheduler, ReduceLROnPlateau):
            self.valueScheduler.step()
        self.valueLosses.append(loss.item())

        if len(self.valueLosses) > 1000:
            self.valueLosses.pop(0)

class AdaptiveQLearner:
    def __init__(self, mode="create", bufferSize=1024, train=None, modelParams={}, load_model=None, load_epoch=None):
        self.train = pd.read_csv(train)
        self.train.feats = self.train.feats.map(lambda xx: frozenset(eval(xx)))
        self.train.source_feats = self.train.source_feats.map(eval)

        self.trainKeys = list(self.train.groupby(["lemma", "form", "feats"]))
        np.random.shuffle(self.trainKeys)
        self.keyIter = iter(self.trainKeys)
        for key, block in self.trainKeys:
            #avoid negative response times
            block.loc[block["response_time"] < 0, "response_time"] = 1
            block.sort_values("response_time", inplace=True)

        if mode == "create":
            self.bufferSize = bufferSize

            self.dataHandler = EditHandler(mode=mode, train=self.train) #DataHandler(mode=mode, train=self.train)
            self.modelParams = modelParams
            self.model = Model(mode=mode, 
                               n_actions=2, 
                               src_vocab_size=self.dataHandler.sourceVocabSize(),
                               trg_vocab_size=self.dataHandler.targVocabSize(),
                               data=self.dataHandler,
                               **modelParams)

            self.simulator = Simulator(dataHandler=self.dataHandler, model=self.model, train=self.train)
        else:
            assert(mode == "load" and load_model != None)
            fSettings = f"{load_model}-.settings"
            with open(fSettings, "rb") as fh:
                settings = pickle.load(fh)
            
            self.bufferSize = settings["bufferSize"]
            self.dataHandler = EditHandler(mode=mode, settings=settings) #DataHandler(mode=mode, settings=settings)
            self.modelParams = settings["modelParams"]
            modelParams["load_model"] = f"{load_model}-{load_epoch}-"
            self.model = Model(mode=mode,
                               src_vocab_size=self.dataHandler.sourceVocabSize(),
                               trg_vocab_size=self.dataHandler.targVocabSize(),
                               data=self.dataHandler,
                               **modelParams)

            self.simulator = Simulator(dataHandler=self.dataHandler, model=self.model, train=self.train)

    def epoch(self):
        self.fillBuffer()
        self.learn()

    def fillBuffer(self):
        self.stateBuffer = None
        self.stats = []
        while self.stateBuffer is None or len(self.stateBuffer) < self.bufferSize:
            block = self.sampleBlock()
            sim, stts = self.simulator.simulate(block)
            self.stats.append(stts)

            if self.stateBuffer is None:
                self.stateBuffer = sim
            else:
                self.stateBuffer = pd.concat([self.stateBuffer, sim], ignore_index=True)

            #print("buffer contains", len(self.stateBuffer))

        waits = [xx[0] for xx in self.stats]
        corrs = [xx[1] for xx in self.stats]
        print(f"mean wait {np.mean(waits)}, mean correct {np.mean(corrs)}")

    def printOutputs(self):
        block = self.sampleBlock()
        (lemma, form, feats), _ = block
        print(f"Lemma: {lemma} target form: {form}")
        sim, stts = self.simulator.simulate(block)
        for ind, row in sim.iterrows():
            source, target, _, _ = row["instance"]
            source = self.dataHandler.defeaturize(source)
            print("\t".join([source, target, row["prediction"], str(row["correct"])]))
        print()

    def sampleBlock(self):
        try:
            return next(self.keyIter)
        except StopIteration:
            np.random.shuffle(self.trainKeys)
            self.keyIter = iter(self.trainKeys)
            return next(self.keyIter)

    def learn(self, batchSize=128):
        # print("learning")
        batches = len(self.stateBuffer) // batchSize
        indices = np.arange(0, len(self.stateBuffer))
        np.random.shuffle(indices)

        # print(self.stateBuffer)

        norm = self.stateBuffer["pr_state"].sum()
        self.stateBuffer.loc[:, ["pr_state"]] /= norm

        for batch in np.arange(0, batches * batchSize, batchSize):
            self.learnStrings(batchSize)
            batchInds = indices[batch : batch + batchSize]
            self.learnValues(batchInds)

        print("Mean string loss:", np.mean(self.model.stringLosses))
        print("Mean value loss:", np.mean(self.model.valueLosses))

    def learnStrings(self, batchSize):
        batch = self.sampleStringBatch(batchSize)
        #print("sampled batch for string learning:", batch)
        instances = self.stateBuffer.loc[batch, "instance"]
        tensors = self.dataHandler.instancesToTensors(instances)
        self.model.trainStringBatch(tensors)

    def sampleStringBatch(self, batchSize):
        #uniform weighting of states?
        prs = self.stateBuffer["pr_state"]
        #inds = np.random.choice(len(prs), replace=False, p=prs, size=batchSize)
        inds = np.random.choice(len(prs), replace=False, size=batchSize)
        return inds

    def learnValues(self, batch):
        instances = self.stateBuffer.loc[batch, "instance"]
        tensors = self.dataHandler.instancesToTensors(instances)
        (src, srcMask, trg, trgMask) = tensors
        if self.modelParams["value_mode"] == "regress":
            reward = self.simulator.normalizeReward(self.stateBuffer.loc[batch, ["reward_stop", "reward_wait"]])
        else:
            reward = self.simulator.actionVector(self.stateBuffer.loc[batch, ["reward_stop", "reward_wait"]])
        reward, rewardMask = self.dataHandler.rewardToTensors(reward)
        tensors = (src, srcMask, reward, rewardMask)
        self.model.trainValueBatch(tensors)

    def writeSettings(self, fstem):
        fOut = f"{fstem}.settings"
        settings = {
            "modelParams" : self.modelParams,
            "bufferSize" : self.bufferSize,
            "sourceVocab" : self.dataHandler.sourceC2I,
            "targVocab" : self.dataHandler.targC2I,
            "maxLenSrc" : self.dataHandler.maxLenSrc,
            "maxLenTarg" : self.dataHandler.maxLenTarg,
            "featToChar" : self.dataHandler.featToChar
        }
        with open(fOut, "wb") as fh:
            pickle.dump(settings, fh)            

    def writeParams(self, fstem):
        fOut = f"{fstem}.params.string"
        torch.save(self.model.stringTransformer, fOut)
        fOut = f"{fstem}.params.value"
        torch.save(self.model.valueTransformer, fOut)

if __name__ == '__main__':
    args = parseArgs()
    dataDir = Path(args.project) / "rl/dataset/"
    neuralDir = Path(args.project) / "neural-transducer/data/reinf_inst"

    checkpoints = Path(args.project) / "neural-transducer/checkpoints"

    cumulative = (not args.noncumulative and not args.single_source)
    language = args.language

    checkpoint = checkpoints / language
    os.makedirs(checkpoint, exist_ok=True)

    warnings.filterwarnings("ignore")
    torch.set_warn_always(False)

    split = "train"
    dataPath = dataDir / (f"query_{language}_{split}.csv")

    modelParams = {
        "src_nb_layers" : 4,
        "src_hid_size" : 1024,
        "trg_nb_layers" : 4,
        "trg_hid_size" : 1024,
        "embed_dim" : 256,
        "nb_heads" : 4,
        "dropout_p" : .3,
        "label_smooth" : .1,
        "tie_trg_embed" : False,
        "value_mode" : "classify"
    }

    aql = AdaptiveQLearner(mode="create", train=dataPath, modelParams=modelParams)
    aql.writeSettings(checkpoint/f"{language}-")
    for epoch in range(50000):
        print(f"Epoch {epoch}")
        aql.epoch()
        if epoch % 500 == 0:
            aql.printOutputs()

            aql.writeParams(checkpoint/f"{language}-{epoch}-")
