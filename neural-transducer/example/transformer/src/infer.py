"""
train
"""
import os
from functools import partial
import numpy as np

from pathlib import Path

import torch
from tqdm import tqdm

import dataloader
import model
import transformer
import util
from decoding import Decode, get_decode_fn
from trainer import BaseTrainer
import transformer_regressor
from train import *

if __name__ == "__main__":
    trainer = Trainer()
    params = trainer.params
    decode_fn = get_decode_fn(
        params.decode, params.max_decode_len, params.decode_beam_size
    )
    src_c2i = None
    attr_c2i = None

    load_model = params.load_previous
    params.load = "smart"
    start_epoch = trainer.smart_load_model(load_model) + 1
    prevM = trainer.model
    src_c2i = prevM.src_c2i
    attr_c2i = prevM.attr_c2i
    print("Forcing vocabulary:", src_c2i)
    trainer.model = None

    trainer.load_data(params.dataset, params.train, params.dev, params.test,
                      src_c2i=src_c2i, attr_c2i=attr_c2i)
    trainer.setup_evalutator()

    start_epoch = trainer.smart_load_model(load_model) + 1

    trainer.setup_training()
    trainer.load_training(load_model)
    trainer.models = [] #prevent the trainer from purging the previous models?
    print(params.dev)
    dname = Path(params.dev[0]).name
    print("try to create", f"{dname}.decode")
    results = trainer.decode("dev", 32, f"{dname}.decode", decode_fn)
    #trainer.evaluate("dev", params.bs, 0, decode_fn)

