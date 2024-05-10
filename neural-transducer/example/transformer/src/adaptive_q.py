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
from simulators import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, mode="create", train=None, cumulative=False, cutoff=600, settings=None):
        super(EditHandler, self).__init__(mode, train, cumulative, cutoff, settings)
        self.analyses = {}
        self.diffs = {}
    
    def clearCache(self):
        self.analyses = {}
        self.diffs = {}

    def readCharset(self, train):
        super().readCharset(train)
        for sym in ["0", "1", "2", "-0", "-1", "-2"]:
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
            chrs = [xx for xx in re.split("((?:<[^>]*>)|(?:[+-]?.))", si) if xx != ""]
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
                                            self.diffs)
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
                 batch_size=32,
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
        self.batch_size = batch_size

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
        #     print("device check", ind, tns.get_device(), "shape", tns.shape)

        nn = tensors[0].shape[1]
        batchSize = self.batch_size
        batches = np.ceil(nn / batchSize)
        # print("dividing up", nn, "data points into", batches, "batches")

        allDecs = []
        for batch in np.arange(0, batches * batchSize, batchSize):
            batch = int(batch)
            # print("batch goes from", batch, "to", batch + batchSize)
            tns = [ti[:, batch : batch + batchSize] for ti in tensors]
            # for ind, ti in enumerate(tns):
            #     print("\tbatch device check", ind, ti.get_device(), "shape", ti.shape)
            
            preds = self.stringTransformer.forward(*tns)

            #seq length x instances x charset
            # print("shape of preds", preds.shape)
            preds = preds.transpose(0, 1)
            ams = np.argmax(preds.cpu().detach().numpy(), axis=-1)
            decs = self.data.decode(ams, self.data.targC2I)
            allDecs += decs

        return allDecs

    def valuePredictions(self, tensors):
        (src, srcMask, _, _) = tensors
        trg = torch.ones((1, src.shape[1]), dtype=torch.long).to(DEVICE)
        trgMask = torch.ones((1, src.shape[1])).to(DEVICE)

        tensors = (src, srcMask, trg, trgMask)

        nn = tensors[0].shape[1]
        batchSize = self.batch_size
        batches = np.ceil(nn / batchSize)

        allPreds = []
        for batch in np.arange(0, batches * batchSize, batchSize):
            batch = int(batch)
            # print("batch goes from", batch, "to", batch + batchSize)
            tns = [ti[:, batch : batch + batchSize] for ti in tensors]
            preds = self.valueTransformer.forward(*tns)
            #squeeze sequence dimension yielding batch x actions
            preds = preds.squeeze(dim=0).cpu().detach().numpy()
            #print("preds", preds.shape)
            #print(preds)
            allPreds.append(preds)

        allPreds = np.concatenate(allPreds, axis=0)

        return allPreds

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
    def __init__(self, mode="create", train=None, settings={}, load_model=None, load_epoch=None):
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
            self.bufferSize = settings["buffer_size"]
            self.nSources = settings["n_sources"]
            self.nExplore = settings["n_explore"]

            self.dataHandler = EditHandler(mode=mode, train=self.train) #DataHandler(mode=mode, train=self.train)
            self.settings = settings
            self.model = Model(mode=mode, 
                               n_actions=1 + self.nSources, 
                               src_vocab_size=self.dataHandler.sourceVocabSize(),
                               trg_vocab_size=self.dataHandler.targVocabSize(),
                               data=self.dataHandler,
                               **settings)

            if self.nSources > 1:
                self.simulator = StochasticMultisourceSimulator(dataHandler=self.dataHandler, model=self.model, 
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
            
            self.settings = settings["settings"]
            self.bufferSize = self.settings["buffer_size"]
            self.nSources = self.settings["n_sources"]
            self.nExplore = self.settings["n_explore"]

            self.dataHandler = EditHandler(mode=mode, settings=settings) #DataHandler(mode=mode, settings=settings)
            self.settings["load_model"] = f"{load_model}-{load_epoch}-"
            self.model = Model(mode=mode,
                               src_vocab_size=self.dataHandler.sourceVocabSize(),
                               trg_vocab_size=self.dataHandler.targVocabSize(),
                               data=self.dataHandler,
                               **self.settings)

            if self.nSources > 1:
                self.simulator = StochasticMultisourceSimulator(dataHandler=self.dataHandler, model=self.model, train=self.train,
                                                                nSources=self.nSources, nExplore=self.nExplore)
            else:
                self.simulator = Simulator(dataHandler=self.dataHandler, model=self.model, train=self.train)

    def epoch(self):
        self.fillBuffer()
        for ii in range(self.settings["epochs_per_buffer"]):
            self.learn()
        self.dataHandler.clearCache()

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
            np.random.shuffle(self.trainKeys)
            self.keyIter = iter(self.trainKeys)
            return next(self.keyIter)

    def learn(self):
        # print("learning")
        batchSize = self.settings["batch_size"]
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
        instances = self.stateBuffer.loc[batch, "value_instance"]
        tensors = self.dataHandler.instancesToTensors(instances)
        (src, srcMask, trg, trgMask) = tensors

        rcs = ["reward_stop", "reward_wait"]
        if self.nSources == 2:
            rcs = ["reward_stop", "reward_wait", "reward_store"]

        if self.settings["value_mode"] == "regress":
            reward = self.simulator.normalizeReward(self.stateBuffer.loc[batch, rcs])
        else:
            reward = self.simulator.actionVector(self.stateBuffer.loc[batch, rcs])
        reward, rewardMask = self.dataHandler.rewardToTensors(reward)
        tensors = (src, srcMask, reward, rewardMask)
        self.model.trainValueBatch(tensors)

    def writeSettings(self, fstem):
        fOut = f"{fstem}.settings"
        settings = {
            "settings" : self.settings,
            "sourceVocab" : self.dataHandler.sourceC2I,
            "targVocab" : self.dataHandler.targC2I,
            "maxLenSrc" : self.dataHandler.maxLenSrc,
            "maxLenTarg" : self.dataHandler.maxLenTarg,
            "featToChar" : self.dataHandler.featToChar,
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
        "batch_size" : 768,
        "n_sources" : 2,
        "buffer_size" : 1024,
        "n_explore" : 3,
        "epochs_per_buffer" : 1
    }

    aql = AdaptiveQLearner(mode="create", train=dataPath, settings=settings)
    aql.writeSettings(checkpoint/f"{language}-")
    for epoch in range(50000):
        print(f"Epoch {epoch}")
        aql.epoch()
        if epoch % 500 == 0:
            aql.printOutputs()

            aql.writeParams(checkpoint/f"{language}-{epoch}-")
