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
from decoding import decode_greedy_transformer
import util
from transformer_adaptors import *
from nn_components import *
from improved_value_estimator import *

def softmax(logits):
    return F.softmax(torch.tensor(logits)).numpy()

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
                 inference_batch_size=32,
                 value_mode="regress",
                 value_predictor="tied_transformer",
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
        self.batch_size = inference_batch_size
        self.value_predictor = value_predictor

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
                self.valueModel = TransformerRegressorFromLayers(encoder=self.encoder,
                                                                 src_embed=self.src_embed,
                                                                 decoder=self.regressor_decoder,
                                                                 trg_embed=self.trg_embed,
                                                                 position_embed=self.position_embed,
                                                                 final_out=self.r_final_out,
                                                                 dropout=self.dropout,
                                                                 embed_scale=self.embed_scale)
            elif self.value_mode == "classify":
                if self.value_predictor == "tied_transformer":
                    self.valueModel = TransformerClassifierFromLayers(encoder=self.encoder,
                                                                      src_embed=self.src_embed,
                                                                      decoder=self.regressor_decoder,
                                                                      trg_embed=self.trg_embed,
                                                                      position_embed=self.position_embed,
                                                                      final_out=self.r_final_out,
                                                                      dropout=self.dropout,
                                                                      embed_scale=self.embed_scale)
                elif self.value_predictor == "conv":
                    self.valueModel = ConvClassifier(nSyms=src_vocab_size,
                                                     strLen=self.data.maxLenSrc,
                                                     nFeats=src_vocab_size, #this is fake but needed for indices
                                                     nEmbed=128,
                                                     nCharEmbed=8,
                                                     dropoutP=dropout_p)
                elif self.value_predictor == "untied_transformer":
                    self.valueEncoder = nn.TransformerEncoder(
                        encoder_layer, num_layers=src_nb_layers, norm=nn.LayerNorm(embed_dim)
                    )
                    self.valueModel = TransformerClassifierFromLayers(encoder=self.valueEncoder,
                                                                      src_embed=self.src_embed,
                                                                      decoder=self.regressor_decoder,
                                                                      trg_embed=self.trg_embed,
                                                                      position_embed=self.position_embed,
                                                                      final_out=self.r_final_out,
                                                                      dropout=self.dropout,
                                                                      embed_scale=self.embed_scale)
                elif self.value_predictor == "multipart_expectation":
                    self.valueModel = ValueEstimator(self.data, nSyms=self.data.sourceVocabSize(),
                                                     strLen=self.data.maxLenSrc,
                                                     nFeats=self.data.sourceVocabSize(),
                                                     nEmbed=128,
                                                     nCharEmbed=8,
                                                     dropoutP=self.dropout_p,
                                                     max_norm=self.max_norm)
                    self.valueModel.correctnessPredictor.to(DEVICE)
                    self.valueModel.waitValuePredictor.to(DEVICE)
                    self.valueModel.stopValuePredictor.to(DEVICE)
                    self.valueOptimizer, self.valueScheduler = self.valueModel.optimizers(lr, (beta1, beta2), warmup)

            if self.value_predictor != "multipart_expectation":
                self.valueModel.to(DEVICE)
    
                self.valueOptimizer = torch.optim.Adam(
                    self.valueModel.parameters(), lr, betas=(beta1, beta2)
                )
    
                self.valueScheduler = util.WarmupInverseSquareRootSchedule(
                    self.valueOptimizer, warmup
                )
                
        else:
            assert(mode == "load" and kwargs["load_model"] != None)
            loadf = kwargs["load_model"]
            self.stringTransformer = torch.load(f"{loadf}.params.string", map_location=DEVICE)
            if self.value_predictor != "multipart_expectation":
                self.valueModel = torch.load(f"{loadf}.params.value", map_location=DEVICE)
            else:
                self.valueModel = ValueEstimator(self.data, nSyms=self.data.sourceVocabSize(),
                                                 strLen=self.data.maxLenSrc,
                                                 nFeats=self.data.sourceVocabSize(),
                                                 nEmbed=128,
                                                 nCharEmbed=8,
                                                 dropoutP=self.dropout_p,
                                                 max_norm=self.max_norm)

                self.valueModel.load(f"{loadf}.params.value")
                self.valueOptimizer, self.valueScheduler = self.valueModel.optimizers(lr, (beta1, beta2), warmup)

            if self.value_predictor == "tied_transformer":
                self.valueModel.encoder = self.stringTransformer.encoder
                self.valueModel.src_embed = self.stringTransformer.src_embed

            self.stringTransformer.to(DEVICE)
    
            self.stringOptimizer = torch.optim.Adam(
                    self.stringTransformer.parameters(), lr, betas=(beta1, beta2)
                )
    
            self.stringScheduler = util.WarmupInverseSquareRootSchedule(
                    self.stringOptimizer, warmup
                )

            if self.value_predictor != "multipart_expectation":
                self.valueModel.to(DEVICE)

                self.valueOptimizer = torch.optim.Adam(
                    self.valueModel.parameters(), lr, betas=(beta1, beta2)
                )
    
                self.valueScheduler = util.WarmupInverseSquareRootSchedule(
                    self.valueOptimizer, warmup
                )

    def setDistributions(self, train):
        #precompute anything that's just a static estimate based on the training set
        #we could estimate these with batched averages but why bother
        if self.value_predictor == "multipart_expectation":
            self.valueModel.setDistribution(train)

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

    def decoderStringPredictions(self, tensors):
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

            (srcTens, srcMask, targTens, targMask) = tns
            decode_fn = decode_greedy_transformer
            preds, _ = decode_fn(self.stringTransformer, srcTens, srcMask)

            # print("decoder preds", preds)

            #seq length x instances
            preds = preds.transpose(0, 1)
            # print("shape of preds", preds.shape)
            preds = preds.detach().cpu().numpy()
            decs = self.data.decode(preds, self.data.targC2I)
            # print("decs", decs)
            # assert(0)

            allDecs += decs

        return allDecs

    def valuePredictions(self, tensors):
        if self.value_predictor not in ["conv", "multipart_expectation"]:
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
            preds = self.valueModel.forward(*tns)
            #squeeze sequence dimension yielding batch x actions
            if len(preds.shape) == 3:
                preds = preds.squeeze(dim=0).cpu().detach().numpy()
            #print("preds", preds.shape)
            #print(preds)
            allPreds.append(preds)

        allPreds = np.concatenate(allPreds, axis=0)
        if len(allPreds.shape) == 1:
            allPreds = allPreds[None, :] #add the batch dimension back in to avoid a really annoying bug

        if self.value_predictor != "multipart_expectation":
            allPreds = softmax(allPreds)

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
        if self.value_predictor == "multipart_expectation":
            losses = self.valueModel.learn(tensors, self.valueOptimizer, self.valueScheduler)
            self.valueLosses.append(losses)

            if len(self.valueLosses) > 1000:
                self.valueLosses.pop(0)

            return

        if self.value_predictor == "conv":
            (src, feats, rewards) = tensors
            rewards = torch.permute(rewards, (1, 0))

            out = self.valueModel.forward(src, feats)
            loss = self.valueModel.loss(out, rewards)
        else:
            (src, srcMask, actualTrg, actualTrgMask) = tensors

            trg = torch.ones((1, src.shape[1]), dtype=torch.long).to(DEVICE)
            trgMask = torch.ones((1, src.shape[1])).to(DEVICE)
            tensors = (src, srcMask, trg, trgMask)

            # for ii, ti in enumerate(tensors):
            #     print("shape of tensor", ii, "\t", ti.shape)

            out = self.valueModel.forward(*tensors)
            # print("shape of predictions", out.shape)
            loss = self.valueModel.loss(out, actualTrg, reduction=True)

        self.valueOptimizer.zero_grad()
        loss.backward()

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.valueModel.parameters(), max_norm)

        self.valueOptimizer.step()
        if not isinstance(self.valueScheduler, ReduceLROnPlateau):
            self.valueScheduler.step()
        self.valueLosses.append(loss.item())

        if len(self.valueLosses) > 1000:
            self.valueLosses.pop(0)

class SelectionModel:
    def __init__(self, mode="create",
                 *,
                 src_vocab_size,
                 embed_dim,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.98,
                 inference_batch_size=32,
                 max_norm=0,
                 **kwargs
             ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.embed_dim = embed_dim
        self.batch_size = inference_batch_size
        self.max_norm = max_norm
        self.losses = []

        if mode == "create":
            self.featureSelector = FeatureSelector(self.src_vocab_size, nEmbed=self.embed_dim)
            self.featureSelector.to(DEVICE)
            self.optimizer = torch.optim.Adam(
                    self.featureSelector.parameters(), lr, betas=(beta1, beta2)
                )
        else:
            assert(mode == "load" and kwargs["load_model"] != None)
            loadf = kwargs["load_model"]
            self.featureSelector = torch.load(f"{loadf}.params.map", map_location=DEVICE)
            self.optimizer = torch.optim.Adam(
                    self.featureSelector.parameters(), lr, betas=(beta1, beta2)
                )

    def predictions(self, tensors):
        nn = tensors[0].shape[0]
        batchSize = self.batch_size
        batches = np.ceil(nn / batchSize)
        # print("dividing up", nn, "data points into", batches, "batches")

        allDecs = []
        for batch in np.arange(0, batches * batchSize, batchSize):
            batch = int(batch)
            # print("batch goes from", batch, "to", batch + batchSize)
            tns = [ti[batch : batch + batchSize, :] for ti in tensors]
            # for ind, ti in enumerate(tns):
            #     print("\tbatch device check", ind, ti.get_device(), "shape", ti.shape)
            
            preds = self.featureSelector.forward(*tns)
            # print("shape of preds", preds.shape)
            preds = F.softmax(preds)
            allDecs.append(preds.cpu().detach().numpy())

        allDecs = np.concatenate(allDecs)
        # print("shape of decs", allDecs.shape)

        if len(allDecs.shape) != 2:
            print("bad shape")
            print(nn)
            print(allDecs.shape)
            print([xx.shape for xx in tensors])
        assert(len(allDecs.shape) == 2)
        return allDecs[:, 1]

    def trainBatch(self, src, ys):
        preds = self.featureSelector.forward(*src)
        loss = self.featureSelector.loss(preds, ys)
        self.optimizer.zero_grad()
        loss.backward()

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.mapLayer.parameters(), max_norm)

        self.optimizer.step()
        self.losses.append(loss.item())

        if len(self.losses) > 1000:
            self.losses.pop(0)
