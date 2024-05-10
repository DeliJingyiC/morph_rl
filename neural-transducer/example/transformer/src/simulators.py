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

from transducers import *

def countOnes(target):
    expr = "(?<=[^-])1"
    rec = re.compile(expr)    
    matches = re.findall(rec, target)
    return len(matches)    

def harmonic_mean(values):
    return 1 / np.mean(1 / np.array(list(values)))

def softmax(logits):
    return F.softmax(torch.tensor(logits)).numpy()

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

    def simulate(self, block, train=True):
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
        valInstances = self.data.blockToValueInstances(lemma, form, feats, block, predictions)
        # print("deciphered", predictions)
        self.data.updateDynamicTargets(lemma, form, feats, block, rawPredictions, predictions)
        valTensors = self.data.instancesToTensors(valInstances)
        qpreds = self.model.valuePredictions(valTensors)
        qpreds = pd.DataFrame(qpreds, columns=["reward_stop", "reward_wait"])
        rewards = self.calcRewards(block, instances, predictions)
        rewards["value_instance"] = valInstances
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

        stats = self.evaluatePolicy(rewards, policy="predicted")

        return rewards, stats

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

        source = None
        for step, ri in rewards.iterrows():
            source = tuple(ri["source_lemma"], ri["source_feats"], ri["source_form"])

            if crit(ri):
                stats = {
                    "steps" : step,
                    "correct" : ri["correct"],
                    "stored" : False,
                    "time" : ri["reward_stop"],
                    "sources" : [source],
                }
                return stats

        stats = {
            "steps" : step,
            "correct" : ri["correct"],
            "stored" : False,
            "time" : ri["reward_stop"],
            "sources" : [source],
        }

        return stats

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

class State:
    def __init__(self, inflect, block, row, store, nStore=1):
        self.inflect = inflect
        self.block = block
        self.row = row
        self.store = store
        self.nStore = nStore
        self.strInstance = None
        self.prediction = None
        self.rawPrediction = None
        self.correct = False
        self.valInstance = None
        self.action = "stop"
        self.predictedAction = "stop"
        self.prob = 0
        self.successors = None
        self.values = None
        self.rewardStop = None
        self.rewardWait = None
        self.rewardStore = None
        self.bestReward = None

    def __str__(self):
        newSource = self.block.loc[self.row, ["source_lemma", "source_feats", "source_form"]].to_list()
        return f"{self.row} {newSource} {self.store}"

    def key(self):
        return (self.row, self.store)

    def enumerateSuccessors(self, factory, actions=["wait", "store"]):
        #actions are wait, stop and store
        #stop action has no successors
        if self.successors == None:
            self.successors = {}

        #wait action moves to the next query, if it exists
        if self.row < len(self.block) - 1:
            if "wait" in actions:
                sWait = factory.getState(State(self.inflect, self.block, self.row + 1, self.store, nStore=self.nStore))
                self.successors["wait"] = sWait

            #store action updates the stored content
            #also can't store if you can't wait--- just have to stop
            if "store" in actions or "store_conditional" in actions:
                newSource = self.block.loc[self.row, ["source_lemma", "source_feats", "source_form"]].to_list()
                if newSource[1] != False:
                    newSource[1] = frozenset(newSource[1])
                    newSource = tuple(newSource)
                    if len(self.store) == self.nStore:
                        newStore = self.store[:-1] + (newSource,)
                    else:
                        newStore = self.store[:] + (newSource,)
                    sStore = factory.getState(State(self.inflect, self.block, self.row + 1, newStore, nStore=self.nStore),
                                              create=("store" in actions))
                    if sStore is not None:
                        self.successors["store"] = sStore
    
        for path, si in self.successors.items():
            if si.row <= self.row:
                print("found a row decrease between", self, "and", path, si)
                assert(0)

        return self.successors.values()

    def instance(self, handler):
        if self.strInstance == None:
            currSource = self.block.loc[self.row, ["source_lemma", "source_feats", "source_form"]].to_list()
            sources = (currSource,) + self.store

            iLemma, iForm, iFeats = self.inflect
            if handler.featToChar != None:
                sources = tuple([ (lemma, handler.mapFeatsToChars(feats), form) for (lemma, feats, form) in sources])
                iFeats = handler.mapFeatsToChars(iFeats)

            key = self.inflect + sources
            if key in handler.analyses:
                (inst, diffed) = handler.analyses[key]

            else:
                inst, diffed = handler.createInstance(iLemma, iForm, iFeats, sources)
                handler.analyses[key] = (inst, diffed)

            self.strInstance = inst

        return self.strInstance

    def adaptiveInstance(self, handler, lemma, form, feats, sources):
        i1, di1 = handler.createInstance(lemma, form, feats, sources)
        i2, di2 = handler.createInstance(lemma, form, feats, list(reversed(sources)))
        # print("creating instance", sources)
        # print(i1, countOnes(i1[1]))
        # print(i2, countOnes(i2[1]))
        # print()

        if countOnes(i1[1]) > countOnes(i2[1]):
            return i1, di1
        else:
            return i2, di2

    def valueInstance(self, handler):
        if self.valInstance == None:
            currLemma, currFeats, currForm = self.block.loc[self.row, ["source_lemma", "source_feats", "source_form"]].to_list()
            _, diff = diffExe(currLemma, currForm, handler.diffs)

            iLemma, iForm, iFeats = self.inflect
            if handler.featToChar != None:
                iFeats = handler.mapFeatsToChars(iFeats)
                currFeats = handler.mapFeatsToChars(currFeats)

            sources = [("", currFeats, diff)]

            if self.prediction == None:
                print("no prediction", self.strInstance, self.prediction)
                assert(0)

            inst = handler.writeValues(iFeats, self.prediction, sources)
            inst = inst[:handler.maxLenSrc - 4]
            self.valInstance = (inst, "", "0", len(inst) > handler.cutoff)

            # print(self.valInstance)

        return self.valInstance

    def decipher(self, handler):
        currSource = self.block.loc[self.row, ["source_lemma", "source_feats", "source_form"]].to_list()
        sources = (currSource,) + self.store

        iLemma, iForm, iFeats = self.inflect
        if handler.featToChar != None:
            sources = tuple([ (lemma, handler.mapFeatsToChars(feats), form) for (lemma, feats, form) in sources])
            iFeats = handler.mapFeatsToChars(iFeats)

        key = self.inflect + sources
        (inst, diffedSources) = handler.analyses[key]
            
        if self.rawPrediction != None:
            stringResult = convertFromEdits(self.rawPrediction, iLemma, diffedSources)
        else:
            stringResult = None

        return stringResult

    def calcReward(self, worst):
        self.correct = (self.prediction == self.block.loc[self.row, "form"])
        if self.correct:
            self.rewardStop = -self.block.loc[self.row, "response_time"]
        else:
            self.rewardStop = worst

        self.rewardWait = worst
        self.rewardStore = worst
        for path, si in self.successors.items():
            if path == "wait":
                if si.bestReward == None:
                    print("wait continuation leads to uninitialized")
                    print(self)
                    print(si)
                    print(si.values)
                    assert(0)

                self.rewardWait = si.bestReward
            elif path == "store":
                if si.bestReward == None:
                    print("store continuation leads to uninitialized")
                    print(self)
                    print(si)
                    print(si.values)
                    assert(0)

                self.rewardStore = si.bestReward

        self.bestReward = max(self.rewardStop, self.rewardWait, self.rewardStore)

    def calcPolicy(self):
        # print("calculating policy at", self, "val inst", self.valInstance, "with values", self.values)

        best = self.rewardStop
        self.action = "stop"
        if self.rewardStore >= best and "store" in self.successors:
            best = self.rewardStore
            self.action = "store"
        if self.rewardWait >= best and "wait" in self.successors:
            best = self.rewardWait
            self.action = "wait"

        self.predictedAction = "stop"
        if self.values[2] >= self.values[0] and "store" in self.successors:
            self.predictedAction = "store"
        if self.values[1] >= max(self.values[0], self.values[2]) and "wait" in self.successors:
            self.predictedAction = "wait"        

    def evaluatePolicy(self, policy, steps):
        if policy == "stop":
            return self, steps
        elif policy == "wait":
            if "wait" in self.successors:
                return self.successors["wait"].evaluatePolicy(policy, steps + 1)
            else:
                return self, steps
        elif policy == "optimal":
            if self.action == "stop":
                return self, steps

            # print(self, list(self.successors.keys()))
            return self.successors[self.action].evaluatePolicy(policy, steps + 1)
        elif policy == "predicted":
            if self.predictedAction == "stop":
                return self, steps

            return self.successors[self.predictedAction].evaluatePolicy(policy, steps + 1)
        else:
            assert(0)

    # def calcProbs(self, pr):
    #     if len(self.successors) == 0:
    #         self.prob = pr

    #     pStop, pWait, pStore = softmax(self.values)
    #     self.prob = pr * pStop

    #     if store not in self.successors:
    #         pWait += pStore

    #     for path, si in self.successors.items():
    #         if path == "store":
    #             si.calcProbs(pStore)
    #         elif path == "wait":
    #             si.calcProbs(pWait)

    def check(self, seen, stab):
        if self in seen:
            print("loop observed")
            print(self)
            assert(0)

        if self not in stab:
            print("state not in the table")
            print(self)
            assert(0)            

        ancestors = seen.union([self])
        for path, si in self.successors.items():
            print("->", path, si)
            si.check(ancestors, stab)

class MultisourceSimulator(Simulator):
    def __init__(self, dataHandler, model, train, nSources=2):
        super(MultisourceSimulator, self).__init__(dataHandler, model, train)
        self.nSources = nSources

    def simulate(self, block, train=True):
        (lemma, form, feats), block = block
        block = block.reset_index()

        # print("inflecting", lemma, form, feats)
        # print("building states")

        self.buildStates((lemma, form, feats), block)

        # print("created", len(self.states), "states")

        stateToInst = { state.key() : state.instance(self.data) for state in self.states.values() }
        instances = list(set(stateToInst.values()))

        # print("generated", len(instances), "instances")
        # for state, inst in sorted(stateToInst.items(), key=lambda xx: xx[0].row):
        #     print(state)
        #     print(inst)
        #     print()

        tensors = self.data.instancesToTensors(instances)

        # print("mapped to tensors")

        rawPredictions = self.model.stringPredictions(tensors)
        # print("raw predictions", rawPredictions)
        rawPredictions = self.spacePrunedValues(rawPredictions, instances)

        # print("length of string predictions:", len(rawPredictions))

        #map instances to predictions and then associate each state with its outcome
        instanceToPred = dict(zip(instances, rawPredictions))
        for si in self.states.values():
            instance = stateToInst[si.key()]
            pred = instanceToPred[instance]
            si.rawPrediction = pred
            si.prediction = si.decipher(self.data)
            # print(si, "-->", si.rawPrediction, "-->", si.prediction)

        stateToVal = { state.key() : state.valueInstance(self.data) for state in self.states.values() }
        valInstances = list(set(stateToVal.values()))

        # print("inflecting", lemma, form, feats)
        # print("generated", len(valInstances), "value instances")
        # for state, inst in sorted(stateToVal.items(), key=lambda xx: xx[0].row):
        #     print(state)
        #     print(inst)
        #     print()
        # assert(0)

        #hook for dynamic targets here--- currently not implemented

        valTensors = self.data.instancesToTensors(valInstances)
        qpreds = self.model.valuePredictions(valTensors)

        # print("shape of value predictions:", qpreds.shape)

        valInstanceToPred = dict(zip(valInstances, qpreds))

        # for kk, vv in valInstanceToPred.items():
        #     print(kk, vv)

        for state in sorted(self.states.values(), key=lambda xx: xx.row):
            valInstance = stateToVal[state.key()]
            pred = valInstanceToPred[valInstance]
            state.values = pred

            # print(state, state.values)

        # print("running check")
        # for state in self.states:
        #     for path, si in state.successors.items():
        #         if si.row <= state.row:
        #             print("row decrease", state, "\t", si)
        #             assert(0)

        #map outputs and predictions back to state space
        state0 = min(self.states.values(), key=lambda xx: xx.row)
        # seen = set()
        # state0.check(seen, self.states)
        # print("checking reward")
        for state in sorted(self.states.values(), key=lambda xx: xx.row, reverse=True):
            state.calcReward(self.worst)
        # print("checking policy")
        for state in sorted(self.states.values(), key=lambda xx: xx.row, reverse=True):
            state.calcPolicy()

        # for state, inst in sorted(stateToVal.items(), key=lambda xx: xx[0].row):
        #     print(state, len(state.successors), state.rewardStop, state.rewardWait, state.rewardStore, state.bestReward)
        #     print()
        # assert(0)

        #format the whole thing as a dataframe
        rewardDF = self.statesToDF()

        # print(rewardDF)
        # rewardDF.to_csv("debug.csv")
        # assert(0)

        stats = self.evaluatePolicy(policy="predicted")

        return rewardDF, stats

    def getState(self, state, create=True):
        # print("request for", state.key(), state.key() in self.states)
        if state.key() in self.states:
            return self.states[state.key()]
        else:
            if create:
                self.states[state.key()] = state
                return state

    def buildStates(self, inflect, block, actions=["wait", "store"]):
        state0 = State(inflect, block, 0, tuple(), nStore=(self.nSources - 1))
        self.states = { state0.key() : state0 }
        queue = [state0]
        ctr = 0
        # print("block is", len(block), "rows long")
        # print("expect number of states bounded by", len(block)**2)
        # print(block)

        while queue:
            ctr += 1
            state = queue.pop()
            for sNew in state.enumerateSuccessors(self, actions):
                if sNew.successors == None:
                    queue.append(sNew)

        # print("actually created", len(self.states), "states:")
        # for key, si in sorted(self.states.items(), key=lambda xx: xx[1].row):
        #     print(key, "\t\t", si)
        # assert(0)

    def statesToDF(self):
        data = []
        for si in sorted(self.states.values(), key=lambda xx: xx.row):
            distr = softmax(si.values)
            vals = {
                "value_instance" : si.valInstance,
                "instance" : si.strInstance,
                "reward_stop" : si.rewardStop,
                "reward_wait" : si.rewardWait,
                "reward_store" : si.rewardStore,
                "optimal_action" : si.action,
                "pred_reward_stop" : distr[0],
                "pred_reward_wait" : distr[1],
                "pred_reward_store" : distr[2],
                "predicted_action" : si.predictedAction,
                "pr_state" : si.prob,
                "correct" : si.correct,
                "prediction" : si.prediction,
                "raw_prediction" : si.rawPrediction,
            }
            data.append(vals)

        return pd.DataFrame.from_records(data)

    def actionVector(self, data, verbose=False):
        rS = data["reward_stop"].to_numpy(dtype="float32")
        rW = data["reward_wait"].to_numpy(dtype="float32")
        rT = data["reward_store"].to_numpy(dtype="float32")

        opt = np.zeros((rS.shape[0], 3), dtype="float32")
        vv = np.stack((rS, rW, rT), axis=1)
        mx = np.max(vv, axis=1)

        opt[rS == mx, 0] = 1
        opt[rW == mx, 1] = 1
        opt[rT == mx, 2] = 1

        opt /= np.sum(opt, axis=1, keepdims=True)

        #order must match dataframe construction around l65
        return opt
    
    def evaluatePolicy(self, policy):
        state0 = min(self.states.values(), key=lambda xx: xx.row)
        final, steps = state0.evaluatePolicy(policy, 0)
        stats = {
            "steps" : steps,
            "correct" : final.correct,
            "stored" : len(final.store) > 0,
            "time" : final.block.loc[final.row, "response_time"],
            "sources" : [final.block.loc[final.row, ["source_lemma", "source_feats", "source_form"]],
                         final.store]
        }

        return stats
        
class StochasticMultisourceSimulator(MultisourceSimulator):
    def __init__(self, dataHandler, model, train, nSources=2, nExplore=3):
        super(StochasticMultisourceSimulator, self).__init__(dataHandler, model, train, nSources=nSources)
        self.nExplore = nExplore

    def simulate(self, block, train=True):
        (lemma, form, feats), block = block
        block = block.reset_index()

        # print("inflecting", lemma, form, feats)
        # print("building states")

        self.buildWaitStates((lemma, form, feats), block)

        # for key, state in sorted(self.states.items(), key=lambda xx: xx[1].row):
        #     print(state, list(state.successors.keys()))
        # print("\n")
        # print("created", len(self.states), "states")

        self.predictStrings()
        self.predictValues()

        nPrev = len(self.states)
        self.buildStoreStates(train=train)

        # print("built", len(self.states) - nPrev, "new states for second pass")

        # # print("after 2nd pass")
        # # for key, state in sorted(self.states.items(), key=lambda xx: xx[1].row):
        # #     print(state, list(state.successors.keys()))
        # # print("\n")

        stateToInst = { state.key() : state.instance(self.data) for state in self.states.values() }
        instances = list(set(stateToInst.values()))

        # print("generated", len(instances), "instances")
        # for key, inst in sorted(stateToInst.items(), key=lambda xx: xx[0]):
        #     print(key)
        #     print(inst)
        #     print()
        # assert(0)

        self.predictStrings()
        self.predictValues()

        #map outputs and predictions back to state space
        state0 = min(self.states.values(), key=lambda xx: xx.row)
        # seen = set()
        # state0.check(seen, self.states)
        # print("checking reward")
        for state in sorted(self.states.values(), key=lambda xx: xx.row, reverse=True):
            state.calcReward(self.worst)
        # print("checking policy")
        for state in sorted(self.states.values(), key=lambda xx: xx.row, reverse=True):
            state.calcPolicy()

        # print("after rewards/policy")
        # for key, state in sorted(self.states.items(), key=lambda xx: xx[1].row):
        #     print(state, list(state.successors.keys()))
        # print("\n")
        # assert(0)

        #format the whole thing as a dataframe
        rewardDF = self.statesToDF()

        # print(rewardDF)
        # rewardDF.to_csv("debug.csv")
        # assert(0)

        stats = self.evaluatePolicy(policy="predicted")
        cmat = self.actionConfusionMatrix(rewardDF)

        return rewardDF, stats, cmat

    def actionConfusionMatrix(self, df):
        rows = []
        for act in df["optimal_action"].unique():
            sub = df.loc[df["optimal_action"] == act, ["predicted_action"]].value_counts()
            row = { "index" : act }
            rows.append(row)
            for si, ct in sub.items():
                row[si] = ct

        cmat = pd.DataFrame.from_records(rows, index="index")
        print(cmat)

    def buildWaitStates(self, inflect, block):
        self.buildStates(inflect, block, actions=["wait"])

    def predictValues(self):
        unprocessed = [ state for state in self.states.values() if state.values is None ]
        stateToVal = { state.key() : state.valueInstance(self.data) for state in unprocessed }
        valInstances = list(set(stateToVal.values()))

        # print("generated", len(valInstances), "value instances")
        # for state, inst in sorted(stateToVal.items(), key=lambda xx: xx[0].row):
        #     print(state)
        #     print(inst)
        #     print()
        # assert(0)

        #hook for dynamic targets here--- currently not implemented

        valTensors = self.data.instancesToTensors(valInstances)
        qpreds = self.model.valuePredictions(valTensors)

        # print("shape of value predictions:", qpreds.shape)

        valInstanceToPred = dict(zip(valInstances, qpreds))

        # for kk, vv in valInstanceToPred.items():
        #     print(kk, vv)

        for state in unprocessed:
            valInstance = stateToVal[state.key()]
            pred = valInstanceToPred[valInstance]
            state.values = pred

    def buildStoreStates(self, train):
        kk = self.nExplore

        stateVals = {}
        for key, state in self.states.items():
            prob = softmax(state.values)[2] #2 = store action
            interest = self.howInteresting(state)
            stateVals[key] = (prob, interest)

        # print("selecting items out of", len(items))
        # print(items)

        queue = []
        #take the top k
        for key, (prob, interest) in sorted(stateVals.items(), key=lambda it: it[1][0],
                                            reverse=True)[:kk]:
            queue.append(key)

        for key in queue:
            # print("selected", self.states[key], "due to policy", stateVals[key][0])
            del stateVals[key]

        #take the most interesting k
        for key, (prob, interest) in sorted(stateVals.items(), key=lambda it: it[1][1],
                                            reverse=True)[:kk]:
            queue.append(key)

        for key in queue[kk:]:
            # print("selected", self.states[key], "due to interest", stateVals[key][1])
            del stateVals[key]

        #take 1 at random
        for ind in np.random.choice(np.arange(len(queue)), replace=False, size=min(len(queue), 1)):
            queue.append(queue[ind])

        queue = [self.states[key] for key in queue]

        # print("queued", len(queue))
        # for qx in queue:
        #     print("chose to expand", qx)
        #     print(qx.key())
        #     print()

        qNew = []
        for state in queue:
            for sNew in state.enumerateSuccessors(self, actions=["store"]):
                qNew.append(sNew)

        # print("generated", len(qNew))
        # for qx in qNew:
        #     print(qx)
        #     print(qx.key())
        #     print(qx.successors)
        #     print(qx.values)
        #     print()
        # assert(0)

        queue = qNew
        while queue:
            state = queue.pop()
            #note: allow store only if it reaches an existing store state?
            for sNew in state.enumerateSuccessors(self, actions=["wait", "store_conditional"]):
                if sNew.successors == None:
                    queue.append(sNew)

    def predictStrings(self):
        unprocessed = [ state for state in self.states.values() if state.prediction == None ]
        stateToInst = { state.key() : state.instance(self.data) for state in unprocessed }
        instances = list(set(stateToInst.values()))

        # print("generated", len(instances), "instances")
        # for state, inst in sorted(stateToInst.items(), key=lambda xx: xx[0].row):
        #     print(state)
        #     print(inst)
        #     print()

        tensors = self.data.instancesToTensors(instances)

        # print("mapped to tensors")

        rawPredictions = self.model.stringPredictions(tensors)
        # print("raw predictions", rawPredictions)
        rawPredictions = self.spacePrunedValues(rawPredictions, instances)

        # print("length of string predictions:", len(rawPredictions))

        #map instances to predictions and then associate each state with its outcome
        instanceToPred = dict(zip(instances, rawPredictions))
        for si in unprocessed:
            instance = stateToInst[si.key()]
            pred = instanceToPred[instance]
            si.rawPrediction = pred
            si.prediction = si.decipher(self.data)
            # print(si, "-->", si.rawPrediction, "-->", si.prediction)

    def howInteresting(self, state):
        src1, targ1 = state.block.loc[state.row, ["source_lemma", "source_form"]].to_list()
        src, targ, _ = state.inflect

        _, diff1 = diffExe(src1, targ1, self.data.diffs)
        _, resid = diffExe(src, targ, self.data.diffs)

        if len(resid) == 0:
            return 0

        x1 = len(lcs(diff1, resid))
        #words are most interesting to store if they cover part of the target but not all
        frac = x1 / len(resid)
        #need a function on [0, 1] which is 0 at the endpoints and high in the middle
        #sin(pi) = sin(0) = 0; sin(pi * .5) = 1
        return np.sin(np.pi * frac)

class HeuristicMultisourceSimulator(StochasticMultisourceSimulator):
    def __init__(self, dataHandler, model, train, nSources=2, nExplore=3):
        super(HeuristicMultisourceSimulator, self).__init__(dataHandler, model, train, 
                                                            nSources=nSources, nExplore=nExplore)

    def simulate(self, block, train=True):
        (lemma, form, feats), block = block
        block = block.reset_index()

        # print("inflecting", lemma, form, feats)
        # print("building states")

        self.buildStates((lemma, form, feats), block)

        interesting = {}
        for key, state in self.states.items():
            interesting[key] = self.howInteresting(state)

        def sKey(item):
            (key, interest) = item
            return (interest, -key[0])

        mark = []
        for key, interest in sorted(interesting.items(), key=sKey, reverse=True)[:self.nExplore]:
            mark.append(key)

        # for key, interest in sorted(interesting.items(), key=sKey, reverse=True):
        #     state = self.states[key]
        #     print(state, state.strInstance, "\t\t", interest)
        # print("\n")
        # assert(0)

        # print("made", len(self.states))
        # for key, state in sorted(self.states.items(), key=lambda xx: xx[1].row):
        #     print(state, list(state.successors.keys()))
        # print("\n")

        if train:
            # print("keys of marked states")
            # for mi in mark:
            #     print(mi)

            self.pruneStates(mark)

        # print("pruned to", len(self.states))
        # for key, state in sorted(self.states.items(), key=lambda xx: xx[1].row):
        #     print(state, list(state.successors.keys()))
        # print("\n")
        # assert(0)

        self.predictStrings()
        self.predictValues()

        #map outputs and predictions back to state space
        state0 = min(self.states.values(), key=lambda xx: xx.row)
        # seen = set()
        # state0.check(seen, self.states)
        # print("checking reward")
        for state in sorted(self.states.values(), key=lambda xx: xx.row, reverse=True):
            state.calcReward(self.worst)
        # print("checking policy")
        for state in sorted(self.states.values(), key=lambda xx: xx.row, reverse=True):
            state.calcPolicy()

        # print("after rewards/policy")
        # for key, state in sorted(self.states.items(), key=lambda xx: xx[1].row):
        #     print(state, list(state.successors.keys()))
        # print("\n")
        # assert(0)

        #format the whole thing as a dataframe
        rewardDF = self.statesToDF()

        # print(rewardDF)
        # rewardDF.to_csv("debug.csv")
        # assert(0)

        stats = self.evaluatePolicy(policy="predicted")

        return rewardDF, stats

    def boundInteresting(self, state):
        if len(state.store) == 0:
            return 0

        src1, targ1 = state.block.loc[state.row, ["source_lemma", "source_form"]].to_list()
        src2, _, targ2 = state.store[0]
        src, targ, _ = state.inflect

        _, diff1 = diffExe(src1, targ1, self.data.diffs)
        _, diff2 = diffExe(src2, targ2, self.data.diffs)
        _, resid = diffExe(src, targ, self.data.diffs)

        # print("bound:")
        # print("\t", src1, targ1, diff1)
        # print("\t", src2, targ2, diff2)
        # print("\t", src, targ, resid)

        x1 = len(lcs(diff1, resid))
        x2 = len(lcs(diff2, resid))

        # print("\t", x1, x2)
        return harmonic_mean([x1, x2])

    def howInteresting(self, state):
        bound = self.boundInteresting(state)
        if bound <= 2:
            return 0

        (instance, target, _, _) = state.instance(self.data)
        nSym = {}
        for sym in range(1, self.nSources + 1):
            expr = f"(?<=[^-]){sym}"
            rec = re.compile(expr)
            matches = re.findall(rec, target)
            nSym[sym] = len(matches)

        return harmonic_mean(nSym.values())

    def pruneStates(self, keep):
        keep = set(keep)

        def retain(state):
            if state.key() in keep:
                return True

            if len(state.store) == 0:
                keep.add(state.key())
                return True

            good = False

            for path, nxt in state.successors.items():
                if path == "wait" or len(state.store) == 0:
                    good = good or retain(nxt)

            if good:
                keep.add(state.key())
                return True

            return False

        retained = {}
        for key, si in self.states.items():
            if retain(si):
                retained[key] = si

        self.states = retained

        for key, si in self.states.items():
            rem = []
            for path, nxt in si.successors.items():
                if nxt.key() not in self.states:
                    rem.append(path)

            for key in rem:
                del si.successors[key]

