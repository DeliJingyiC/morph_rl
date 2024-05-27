import sys
import os
import re
import functools
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
from models import *
from simulators import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataHandler:
    def __init__(self, mode="create", train=None, cumulative=False, cutoff=600, settings=None, harmony=False):
        self.startCP=0x2100
        self.cumulative = cumulative
        self.cutoff = cutoff
        self.harmony = harmony

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
        chs = [[i2c.get(ind, "<UNK>") for ind in seq] for seq in inds]

        # debugging code only--- legit models do produce UNK early in training but trained models should learn not to
        # if any(["<UNK>" in ch for ch in chs]):
        #     print("i2c", i2c)
        #     print("c2i", c2i)
        #     print(inds.tolist())
        #     print(chs)
        #     assert(0)
        #print(chs)

        chs = [self.stripEOS(ch) for ch in chs]
        return ["".join(ch) for ch in chs]

    def defeaturize(self, chrs):
        f2name = dict({ vv : kk for (kk, vv) in self.featToChar.items() })
        chrs = [f2name.get(ch, ch) for ch in sorted(list(chrs))]
        return ";".join(chrs)

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

    def valueInstancesToTensors(self, instances):
        sources = [src for src, fs, _, prune in instances if not prune]
        feats = [fs for src, fs, _, prune in instances if not prune]

        srcInds = self.padAndEncode(sources, self.sourceC2I)
        featsV = self.multihot(feats, self.sourceC2I)

        #debug block in case strlengths were calculated wrongly
        for si, ii in zip(srcInds, sources):
            if len(si) >= self.maxLenSrc:
                print("long seq:", len(si), len(ii), self.maxLenSrc)
                print(si, ii)

        srcTensor, srcMask = self.listToTensor(srcInds, self.maxLenSrc)
        featTensor = self.multihotToTensor(featsV)

        #transformer wants this backwards, conv net has the normal order
        srcTensor = torch.permute(srcTensor, (1, 0))

        return srcTensor, featTensor

    def multihot(self, strings, c2i):
        res = []
        unk = c2i["<UNK>"]
        for si in strings:
            inds = [c2i.get(ci, unk) for ci in si]
            res.append(inds)

        return res

    def multihotToTensor(self, inds):
        maxLen = self.sourceVocabSize()
        data = np.zeros((len(inds), maxLen), dtype="float32")
        # print("DEVICE is", DEVICE)
        # for dev in range(torch.cuda.device_count()):
        #     print(torch.cuda.get_device_name(dev))
        # print("device check for new tensor", data.get_device())
        for ii, seq in enumerate(inds):
            for si in seq:
                data[ii, si] = 1

        data = torch.from_numpy(data).to(DEVICE)

        return data

    def selectionInstancesToTensors(self, instances):
        sources = [src for src, store, trg in instances]
        stores = [store for src, store, trg in instances]
        targs = [trg for src, store, trg in instances]

        srcInds = self.multihot(sources, self.sourceC2I)
        storeInds = self.multihot(stores, self.sourceC2I)
        targInds = self.multihot(targs, self.sourceC2I)

        srcTensor = self.multihotToTensor(srcInds)
        storeTensor = self.multihotToTensor(storeInds)
        targTensor = self.multihotToTensor(targInds)

        return srcTensor, storeTensor, targTensor

    def processFeats(self, feats, lemma, featsep):
        # return featsep.join([xx for xx in feats])
        if feats == False:
            return ""
        return featsep.join([xx for xx in sorted(feats) if xx != lemma])

    def writeValues(self, feats, product, sources,
             featsep=";", lsep=".", infsep=">", instsep=":"):
        sourceStrings = [
            f"{lsep}{self.processFeats(srcFeats, srcLemma, featsep)}{infsep}{srcForm}" for
            (srcLemma, srcFeats, srcForm)
            in sources if srcFeats is not False]
        fullStr = f"{lsep}{self.processFeats(feats, '', featsep)}{infsep}{product}{instsep}{instsep.join(sourceStrings)}"
        return fullStr

    def writeRow(self, lemma, feats, sources,
             featsep=";", lsep=".", infsep=">", instsep=":"):
        sourceStrings = [
            f"{srcLemma}{lsep}{self.processFeats(srcFeats, srcLemma, featsep)}{infsep}{srcForm}" for
            (srcLemma, srcFeats, srcForm)
            in sources if srcFeats is not False]
        fullStr = f"{lemma}{lsep}{self.processFeats(feats, lemma, featsep)}{instsep}{instsep.join(sourceStrings)}"
        return fullStr

    def mapFeatsToChars(self, feats):
        if feats == False:
            return False
        res = []
        for feat in feats:
            if feat in self.featToChar:
                res.append(self.featToChar[feat])

        return tuple(sorted(res))

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

    def blockToValueInstances(self, lemma, trg, feats, block, predictions):
        if self.featToChar != None:
            feats = self.mapFeatsToChars(feats)

        instances = []
        for (index, source), pred in zip(block.iterrows(), predictions):
            sources = [("", "", pred)]
            inst = self.writeRow("", feats, sources)
            #apply truncation--- should be harmless, because all truncated
            #instances will be wrong
            inst = inst[:self.maxLenSrc - 4]
            instances.append((inst, "", "0", len(inst) >= self.cutoff))

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
    def __init__(self, mode="create", train=None, cumulative=False, cutoff=600, settings=None, harmony=False,
                 cacheDump=None):
        super(EditHandler, self).__init__(mode, train, cumulative, cutoff, settings)
        self.analyses = {}
        self.diffs = {}
        self.harmony = harmony

        if cacheDump is not None:
            self.readCache(cacheDump)

    def clearCache(self):
        pass
        # print("should I clear the cache?")
        # print("size of analyses:", util.sizeof_fmt(util.rgetsizeof(self.analyses)), len(self.analyses))
        # print("size of diffs:", util.sizeof_fmt(util.rgetsizeof(self.diffs)), len(self.diffs))
        #self.analyses = {}
        #self.diffs = {}

    def writeCache(self, ofh):
        pickle.dump([self.analyses, self.diffs], ofh)

    def readCache(self, ifn):
        for cacheDump in os.listdir(ifn):
            with open(Path(ifn) / cacheDump, "rb") as ifile:
                [anas, diffs] = pickle.load(ifile)
                self.analyses.update(anas)
                self.diffs.update(diffs)
        
        print("Successfully loaded", len(self.analyses), len(self.diffs), "items from caches")

    def readCharset(self, train):
        super().readCharset(train)
        for sym in ["0", "1", "2", "-0", "-1", "-2", "@0", "@1", "@2"]:
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
            chrs = [xx for xx in re.split("((?:<[^>]*>)|(?:[@+-]?.))", si) if xx != ""]
            si = ["<BOS>"] + chrs + ["<EOS>"]
            inds = [c2i.get(ci, unk) for ci in si]
            res.append(inds)

            # this can't be on in production code because value instances contain
            # output from the transformer, which can contain UNK
            # if unk in inds:
            #     print("created unknown in string encoding:", si)
            #     print(chrs)
            #     print(inds)
            #     print(c2i)
            #     assert(0)

            # print("indices for", si)
            # print("split into", chrs)
            # print("indices", inds)
            # print()

        return res

    def createInstance(self, lemma, trg, feats, sources):
        analysis, diffedExes = formatInputs(lemma, trg, 
                                            [(srcL, srcT) for (srcL, srcF, srcT) in sources],
                                            self.diffs,
                                            harmony=self.harmony)
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
                stringResult = convertFromEdits(pred, lemma, diffedSources, harmony=self.harmony)
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

class AdaptiveQLearner:
    def __init__(self, mode="create", train=None, settings={}, load_model=None, load_epoch=None):
        self.train = pd.read_csv(train)
        self.train.feats = self.train.feats.map(lambda xx: frozenset(eval(xx)))
        self.train.source_feats = self.train.source_feats.map(eval)
        self.batches = 0
        self.passes = 0

        self.trainKeys = list(self.train.groupby(["lemma", "form", "feats"]))
        np.random.shuffle(self.trainKeys)
        self.keyIter = iter(self.trainKeys)
        for key, block in self.trainKeys:
            #avoid negative response times
            block.loc[block["response_time"] < 0, "response_time"] = 1
            block.sort_values("response_time", inplace=True)

        if mode == "create":
            self.bufferSize = settings["buffer_size"]
            self.nSources = settings["n_sources"]
            self.nExplore = settings["n_explore"]
            cacheDump = settings["aligner_cache"]

            self.dataHandler = EditHandler(mode=mode, train=self.train, harmony=settings["harmony"], cacheDump=cacheDump)
            self.settings = settings
            self.selectionModel = None
            self.model = Model(mode=mode, 
                               n_actions=2,
                               src_vocab_size=self.dataHandler.sourceVocabSize(),
                               trg_vocab_size=self.dataHandler.targVocabSize(),
                               data=self.dataHandler,
                               **settings)

            if self.nSources > 1:
                self.selectionModel = SelectionModel(mode=mode,
                                                     src_vocab_size=self.dataHandler.sourceVocabSize(),
                                                     **settings)

                self.simulator = MemorySelectSimulator(dataHandler=self.dataHandler, model=self.model, selectionModel=self.selectionModel,
                                                                train=self.train, nSources=self.nSources,
                                                                nExplore=self.nExplore)
                self.dataHandler.maxLenSrc *= 2 #look back at this
            else:
                self.simulator = Simulator(dataHandler=self.dataHandler, model=self.model, train=self.train)
        else:
            assert(mode == "load" and load_model != None)
            fSettings = f"{load_model}-.settings"
            with open(fSettings, "rb") as fh:
                settings = pickle.load(fh)

            self.batches = settings["batches"]
            self.passes = settings["passes"]

            self.settings = settings["settings"]
            self.bufferSize = self.settings["buffer_size"]
            self.nSources = self.settings["n_sources"]
            self.nExplore = self.settings["n_explore"]
            self.selectionModel = None

            self.dataHandler = EditHandler(mode=mode, settings=settings, harmony=self.settings["harmony"], cacheDump=None)
            self.settings["load_model"] = f"{load_model}-{load_epoch}-"
            self.model = Model(mode=mode,
                               src_vocab_size=self.dataHandler.sourceVocabSize(),
                               trg_vocab_size=self.dataHandler.targVocabSize(),
                               data=self.dataHandler,
                               **self.settings)

            if self.nSources > 1:
                self.selectionModel = SelectionModel(mode=mode,
                                                     src_vocab_size=self.dataHandler.sourceVocabSize(),
                                                     **self.settings)
                self.simulator = MemorySelectSimulator(dataHandler=self.dataHandler, model=self.model, selectionModel=self.selectionModel,
                                                       train=self.train, nSources=self.nSources, nExplore=self.nExplore)
            else:
                self.simulator = Simulator(dataHandler=self.dataHandler, model=self.model, train=self.train)

    def epoch(self):
        self.fillBuffer()
        for ii in range(self.settings["epochs_per_buffer"]):
            self.learn()
        self.dataHandler.clearCache()
        #sys.exit(1) #place exit statement here when profiling

    def fillBuffer(self):
        self.stateBuffer = None
        self.stats = []
        self.cmats = []

        while self.stateBuffer is None or len(self.stateBuffer) < self.bufferSize:
            block = self.sampleBlock()
            sim, stts, cmat = self.simulator.simulate(block)
            self.stats.append(stts)
            self.cmats.append(cmat)

            if self.stateBuffer is None:
                self.stateBuffer = sim
            else:
                self.stateBuffer = pd.concat([self.stateBuffer, sim], ignore_index=True)

            #print("buffer contains", len(self.stateBuffer))

        waits = [xx["steps"] for xx in self.stats]
        corrs = [xx["correct"] for xx in self.stats]
        if "stored" in self.stats[0]:
            storeds = [xx["stored"] for xx in self.stats]
        else:
            storeds = 0

        print(f"mean wait {np.mean(waits)}, mean correct {np.mean(corrs)} mean stored {np.mean(storeds)}")

        final = self.cmats[0]
        for cm in self.cmats[1:]:
            final = final.add(cm, fill_value=0)
        print(final)

    def printOutputs(self):
        block = self.sampleBlock()
        (lemma, form, feats), _ = block
        print(f"Lemma: {lemma} target form: {form}")
        sim, stts, _ = self.simulator.simulate(block)
        for ind, row in sim.iterrows():
            source, target, _, _ = row["instance"]
            source = self.dataHandler.defeaturize(source)
            print("\t".join([source, target, row["prediction"], str(row["correct"])]))
        print()

    def sampleBlock(self):
        try:
            return next(self.keyIter)
        except StopIteration:
            self.passes += 1
            np.random.shuffle(self.trainKeys)
            self.keyIter = iter(self.trainKeys)
            return next(self.keyIter)

    def learn(self):
        # print("learning")
        batchSize = self.settings["batch_size"]
        strings = self.stateBuffer["instance"].to_list()
        if isinstance(strings[0], list):
            strings = functools.reduce(list.__add__, strings, [])

        #prune the strings we aren't going to run
        strings = [xi for xi in strings if not xi[-1]]

        values = self.stateBuffer["value_instance"]
        reward = self.simulator.actionVector(self.stateBuffer)
        if isinstance(values[0], list):
            values = functools.reduce(list.__add__, values, [])

        #prune the values we aren't going to run
        vz = [(val, rew) for (val, rew) in zip(values, reward) if not val[-1]]
        values = [vi[0] for vi in vz]
        reward = [vi[1] for vi in vz]

        selections = None
        selectionTargets = None
        if "selection_instances" in self.stateBuffer.columns:
            selections, selectionTargets = self.simulator.debufferSelections(self.stateBuffer)

        for stringBatch, valBatch, selBatch in util.batcher([strings, (values, reward), (selections, selectionTargets)], batchSize=batchSize):
            self.learnStrings(stringBatch)
            self.learnValues(*valBatch)
            self.learnSelections(*selBatch)

        print("Mean string loss:", np.mean(self.model.stringLosses))
        print("Mean value loss:", np.mean(self.model.valueLosses))
        if self.selectionModel is not None:
            print("Mean selection loss:", np.mean(self.selectionModel.losses))
        print(self.batches, "batches", self.passes, "complete runs")

    def learnStrings(self, instances):
        if instances is not None:
            self.batches += 1
            tensors = self.dataHandler.instancesToTensors(instances)
            self.model.trainStringBatch(tensors)

    def learnValues(self, instances, reward):
        if instances is None:
            return

        if isinstance(instances[0][0], tuple):
            assert(0), "not implemented anymore"
        else:
            tensors = self.dataHandler.valueInstancesToTensors(instances)
            if self.settings["value_predictor"] == "conv":
                (src, feats) = tensors
                reward, rewardMask = self.dataHandler.rewardToTensors(np.array(reward))
                tensors = (src, feats, reward)
                self.model.trainValueBatch(tensors)
            else:
                (src, srcMask, trg, trgMask) = tensors
                reward, rewardMask = self.dataHandler.rewardToTensors(np.array(reward))
                tensors = (src, srcMask, reward, rewardMask)
                self.model.trainValueBatch(tensors)

    def learnSelections(self, instances, targets):
        if self.selectionModel is None or instances is None:
            return

        tensors = self.dataHandler.selectionInstancesToTensors(instances)
        targets = np.array(targets, dtype="int")

        # print("selection instances", instances)
        # print("selection targets", targets)

        #for integer targets
        #targetTensor = torch.tensor(targets, dtype=torch.long).to(DEVICE)
        targetTensor = torch.tensor(targets, dtype=torch.float).to(DEVICE)
        self.selectionModel.trainBatch(tensors, targetTensor)

    def writeSettings(self, fstem):
        fOut = f"{fstem}.settings"
        settings = {
            "settings" : self.settings,
            "sourceVocab" : self.dataHandler.sourceC2I,
            "targVocab" : self.dataHandler.targC2I,
            "maxLenSrc" : self.dataHandler.maxLenSrc,
            "maxLenTarg" : self.dataHandler.maxLenTarg,
            "featToChar" : self.dataHandler.featToChar,
            "batches" : self.batches,
            "passes" : self.passes,
        }
        with open(fOut, "wb") as fh:
            pickle.dump(settings, fh)            

    def writeParams(self, fstem):
        fOut = f"{fstem}.params.string"
        torch.save(self.model.stringTransformer, fOut)
        fOut = f"{fstem}.params.value"
        torch.save(self.model.valueModel, fOut)
        if self.selectionModel != None:
            fOut = f"{fstem}.params.map"
            torch.save(self.selectionModel.featureSelector, fOut)

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

    settings = {
        "src_nb_layers" : 4,
        "src_hid_size" : 1024,
        "trg_nb_layers" : 4,
        "trg_hid_size" : 1024,
        "embed_dim" : 256,
        "nb_heads" : 4,
        "dropout_p" : .3,
        "label_smooth" : .1,
        "tie_trg_embed" : False,
        "value_mode" : "classify",
        "batch_size" : 128,
        "inference_batch_size" : 768,
        "n_sources" : 2,
        "buffer_size" : 8192,
        "n_explore" : 3,
        "epochs_per_buffer" : 1,
        "tie_value_predictor" : False,
        "value_predictor" : "conv",
        "harmony" : True,
        "aligner_cache" : Path(args.project) / f"neural-transducer/aligner_cache/{split}",
    }

    if args.epoch > 0:
        aql = AdaptiveQLearner(mode="load", train=dataPath, 
                               load_model=checkpoint/language, load_epoch=args.epoch)
    else:
        aql = AdaptiveQLearner(mode="create", train=dataPath, settings=settings)
        aql.writeSettings(checkpoint/f"{language}-")

    for epoch in range(args.epoch, 50000):
        print(f"Epoch {epoch}")
        aql.epoch()
        if epoch % 10 == 0:
            #aql.printOutputs()

            aql.writeParams(checkpoint/f"{language}-{epoch}-")
