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

from nn_components import ConvClassifier
import util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def softmax(logits):
    return F.softmax(torch.tensor(logits)).numpy()

class ValueEstimator:
    def __init__(self, handler,
                 nSyms,
                 strLen,
                 nFeats,
                 nEmbed,
                 nCharEmbed,
                 dropoutP,
                 max_norm,
                 product_inference="composite"):
        self.handler = handler
        self.correctnessPredictor = ConvClassifier(nSyms=nSyms,
                                                   strLen=strLen,
                                                   nFeats=nFeats,
                                                   nEmbed=nEmbed,
                                                   nCharEmbed=nCharEmbed,
                                                   dropoutP=dropoutP)
        self.waitValuePredictor = v2ToNum(nFeats=nFeats, nEmbed=nEmbed, dropoutP=dropoutP)
        self.stopValuePredictor = v2ToNum(nFeats=nFeats, nEmbed=nEmbed, dropoutP=dropoutP)
        self.max_norm = max_norm
        self.copula = EmpiricalCopula()
        self.product_inference = product_inference

    def save(self, stem):
        self.copula.save(f"{stem}.copula")
        torch.save(self.waitValuePredictor, f"{stem}.wvp")
        torch.save(self.stopValuePredictor, f"{stem}.svp")
        torch.save(self.correctnessPredictor, f"{stem}.cp")

    def load(self, stem):
        self.copula.load(f"{stem}.copula")
        self.waitValuePredictor = torch.load(f"{stem}.wvp", map_location=DEVICE)
        self.stopValuePredictor = torch.load(f"{stem}.svp", map_location=DEVICE)
        self.correctnessPredictor = torch.load(f"{stem}.cp", map_location=DEVICE)

    def setDistribution(self, dataset):
        self.worst = dataset["response_time"].max()

        print("initializing copula")
        self.copula.setDistribution(dataset)
        print("test copula")
        for pi in [0, 100, 200, 500, 1000, 1e10]:
            lat = self.copula.toLatent(pi)
            print(pi, lat, self.copula.toData(lat))
        print("---")

    def forward(self, strings, targetFeatures, queryFeatures, rts, verbose=False):
        correct = self.correctnessPredictor.forward(strings, targetFeatures)
        wpred = -self.waitValuePredictor.predict(queryFeatures, targetFeatures)
        correct = F.softmax(correct)

        if self.product_inference == None:
            timeStop = self.stopValuePredictor.predict(queryFeatures, targetFeatures)
        elif self.product_inference == "composite":
            timeStop = correct[:, 0:1] * torch.tensor(-rts).to(DEVICE) + correct[:, 1:] * -self.worst
            overall = torch.cat((timeStop, wpred), axis=1)
        elif self.product_inference == "product_only":
            overall = correct

        return overall.cpu().detach().numpy()

        #wpred = wpred.cpu().detach().numpy()

        # if verbose:
        #     print("wpred:", wpred.shape)
        #     print(wpred)

        #deltaWorst = (self.worst - rts).cpu().detach().numpy()

        # if verbose:
        #     print("deltaw", deltaWorst.shape)
        #     print(deltaWorst)

        #use softmax to transform correctness
        # correct = softmax(correct.cpu().detach().numpy())

        # if verbose:
        #     print("correct", correct.shape)
        #     print(correct)

        #correct[:, 1] is prob of being wrong--- 
        #pr. of being right has relative cost 0
        # timeStop = correct[:, 1:] * deltaWorst
        #timeWait = self.copula.toData(wpred)

        # print("expected time if stop")
        # print(timeStop.shape)
        # print(timeStop)
        # print("expected time if wait")
        # print(timeWait.shape)
        # print(timeWait)

        # overall = np.concatenate((timeStop, timeWait), axis=1)

        # if verbose:
        #     print("overall", overall.shape)
        #     print(overall)

        #expected times are negative
        # return -overall

    def rightOrWrong(self, rewards):
        correct = (rewards[:, 0] > -self.worst).cpu().detach().numpy()
        opt = np.zeros((len(correct), 2), dtype="float32")
        opt[:, 0] = correct
        opt[:, 1] = 1 - correct

        # print(rewards[:, 0])
        # print(self.worst)
        # print("Correctness vector", correct.shape)
        # print(correct)
        # print("opt vector", opt.shape)
        # print(opt)
        # assert(0)

        return opt

    def learn(self, tensors, optimizers, schedulers):
        # for ii, ti in enumerate(tensors):
        #     print(f"tensor {ii} shape: {ti.shape}")

        (strings, targetFeatures, queryFeatures, rts, rewards) = tensors
        rewards = torch.permute(rewards, (1, 0))

        rightWrong = self.rightOrWrong(rewards)
        rightWrong = torch.tensor(rightWrong).to(DEVICE)

        (corrOpt, rOpt, sOpt) = optimizers
        (corrSched, rSched, sSched) = schedulers

        out = self.correctnessPredictor.forward(strings, targetFeatures)
        corrLoss = self.correctnessPredictor.loss(out, rightWrong)

        corrOpt.zero_grad()
        corrLoss.backward()

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.correctnessPredictor.parameters(), max_norm)

        corrOpt.step()
        if not isinstance(corrSched, ReduceLROnPlateau):
            corrSched.step()

        #prune instances for which rt is -1
        rts = rts[:, 0]
        sel = (rts != -1)
        queryFeatures = queryFeatures[sel]
        targetFeatures = targetFeatures[sel]

        # relative = -rewards[sel, 1].detach().cpu() - rts[sel]
        # latent = self.copula.toLatent(relative)

        # print("actual rewards of waiting")
        # print(rewards[sel, 1].tolist())
        # print("current rt")
        # print(rts[sel].tolist())
        # print("relative reward of waiting", relative.shape)
        # print(relative.tolist())
        # print("latent-space")
        # print(latent.tolist())

        wpred = self.waitValuePredictor.predict(queryFeatures, targetFeatures)

        # print("predictions")
        # print(wpred.detach().cpu().numpy().tolist())

        # print("relative copula values", latent.shape)

        # print("info on loss:")
        # print(sel.shape)
        # print(wpred)
        # print(latent)

        # rLoss = self.waitValuePredictor.loss(wpred, torch.tensor(latent).to(DEVICE))
        rLoss = self.waitValuePredictor.loss(wpred, -rewards[sel, 1])

        # print("loss:", rLoss)

        rOpt.zero_grad()
        rLoss.backward()

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.waitValuePredictor.parameters(), max_norm)

        rOpt.step()
        if not isinstance(corrSched, ReduceLROnPlateau):
            rSched.step()

        spred = self.stopValuePredictor.predict(queryFeatures, targetFeatures)
        sLoss = self.stopValuePredictor.loss(spred, torch.tensor(-rts[sel]).to(DEVICE))
        sOpt.zero_grad()
        sLoss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.stopValuePredictor.parameters(), max_norm)

        sOpt.step()
        if not isinstance(corrSched, ReduceLROnPlateau):
            sSched.step()

        return corrLoss.item(), rLoss.item(), sLoss.item()

    def optimizers(self, lr, betas, warmup):
        self.opt1 = torch.optim.Adam(
            self.correctnessPredictor.parameters(), lr, betas=betas,
        )

        self.sched1 = util.WarmupInverseSquareRootSchedule(
            self.opt1, warmup
        )

        self.opt2 = torch.optim.Adam(
            self.waitValuePredictor.parameters(), lr, betas=betas,
        )

        self.sched2 = util.WarmupInverseSquareRootSchedule(
            self.opt2, warmup
        )

        self.opt3 = torch.optim.Adam(
            self.stopValuePredictor.parameters(), lr, betas=betas,
        )

        self.sched3 = util.WarmupInverseSquareRootSchedule(
            self.opt3, warmup
        )

        return (self.opt1, self.opt2, self.opt3), (self.sched1, self.sched2, self.sched3)

class v2ToNum(nn.Module):
    def __init__(self, nFeats, nEmbed, dropoutP):
        super(v2ToNum, self).__init__()
        self.activation = F.relu
        self.dropout = nn.Dropout(p=dropoutP)

        self.featRep1 = nn.Linear(nFeats, nEmbed)
        self.featRep2 = nn.Linear(nFeats, nEmbed)
        self.predecision = nn.Linear(2 * nEmbed, nEmbed)
        self.scale = nn.Linear(nEmbed, 1)

    def forward(self, features1, features2):
        #map features to a vector
        # print("features", features.shape, features.dtype)
        fRep = self.featRep1(features1)
        fRep = self.activation(fRep)
        fRep2 = self.featRep2(features2)
        fRep2 = self.activation(fRep2)

        both = torch.cat([fRep, fRep2], axis=1)
        predec = self.predecision(both)
        predec = self.activation(predec)
        predec = self.dropout(predec)

        value = self.scale(predec)
        return value

    def loss(self, predict, target):
        loss = F.l1_loss(predict, target)

        return loss

    def predict(self, target, sNext):
        return self.forward(target, sNext)

class EmpiricalCopula:
    def __init__(self):
        self.values = []

    def save(self, fstem):
        with open(fstem, "wb") as fh:
            pickle.dump(self.values, fh)

    def load(self, fstem):
        with open(fstem, "rb") as fh:
            self.values = pickle.load(fh)

    def setDistribution(self, dataset):
        if len(self.values) > 0:
            self.nn = self.values.shape[0]
            return #preloaded values

        for nBlock, (key, block) in enumerate(dataset.groupby(["lemma", "form", "feats"])):
            block.sort_values("response_time", inplace=True)
            for ind, row in block.iterrows():
                rt = row["response_time"]
                #self.values.append(rt)
                for ind2, row2 in block.iterrows():
                    if ind2 < ind:
                        rt2 = row2["response_time"]
                        delta = rt - rt2
                        if delta < 0:
                            print(block["response_time"])
                            print(ind, ind2)
                        assert(delta >= 0)
                        self.values.append(delta)
                    else:
                        break

            # print(f"{nBlock} blocks read, {len(self.values)} values")
            if nBlock > 500:
                break

        self.values = np.array(sorted(self.values))
        self.nn = self.values.shape[0]

        for ind in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            print(f"{ind}th percentile of values: {np.percentile(self.values, ind)}")

    def toLatent(self, qs):
        latents = np.searchsorted(self.values, qs)
        return (latents / self.nn).astype("float32")

    def toData(self, qs):
        inds = (qs * self.nn).round().astype("int")
        #print("computed inds", inds)
        inds = np.clip(inds, 0, self.nn - 1)
        return self.values[inds]
