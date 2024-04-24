import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import *
from transformer import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RegressionDataLoader(Seq2SeqDataLoader):
    def read_file(self, file):
        with open(file, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                line = line.strip()
                if not line:
                    continue
                toks = line.split("\t")
                # input(toks)
                if len(toks) != 3:
                    print("WARNING: missing tokens", toks)
                    continue
                lemma, targets, tags = toks
                targets = targets.split(";")
                targets = [float(xx) for xx in targets]
                yield list(lemma), targets, tags.split(";")

    def _iter_helper(self, file):
        for lemma, trg, tags in self.read_file(file):
            # print("coding up", lemma, trg, tags, "with unk=", UNK_IDX)

            src = [self.source_c2i[BOS]]
            for tag in tags:
                src.append(self.attr_c2i.get(tag, UNK_IDX))
            for char in lemma:
                src.append(self.source_c2i.get(char, UNK_IDX))
            src.append(self.source_c2i[EOS])

            yield src, trg

    def build_vocab(self):
        char_set, tag_set = set(), set()
        self.nb_train = 0
        for lemma, trg, tags in self.read_file(self.train_file):
            # print(lemma)
            # print(trg)
            # print(tags)
            # # input()
            nActions = len(trg)
            self.nb_train += 1
            char_set.update(lemma)
            tag_set.update(tags)
        self.nb_dev = sum([1 for _ in self.read_file(self.dev_file)])
        if self.test_file is not None:
            self.nb_test = sum([1 for _ in self.read_file(self.test_file)])
        chars = sorted(list(char_set))
        tags = sorted(list(tag_set))
        self.nb_attr = len(tags)
        source = [PAD, BOS, EOS, UNK] + chars + tags
        target = list(range(nActions)) #placeholder used to determine size of output layer
        return source, target

    def sanity_check(self):
        assert self.source[PAD_IDX] == PAD
        assert self.source[BOS_IDX] == BOS
        assert self.source[EOS_IDX] == EOS
        assert self.source[UNK_IDX] == UNK

    def numeric_to_tensor(self, lst: List[List[float]]):
        #assume that all these lns are the same
        #max_len = max([len(x) for x in lst])
        max_len = len(lst[0])
        data = torch.zeros((max_len, len(lst)), dtype=torch.float)
        for i, seq in tqdm(enumerate(lst), desc="build tensor"):
            data[: len(seq), i] = torch.tensor(seq)
        mask = torch.ones((1, len(lst)), dtype=torch.float)
        return data, mask

    def _batch_sample(self, batch_size, file, shuffle):
        key = self._file_identifier(file)
        # print("key",key)
        # print("self.batch_data",self.batch_data)
        # input()
        if key not in self.batch_data:
            # print("inside the if loop")
            lst = list()
            for src, trg in tqdm(self._iter_helper(file), desc="read file"):
                # print("src",src)
                # print("trg",trg)
                # input()  
                lst.append((src, trg))
            
            src_data, src_mask = self.list_to_tensor([src for src, _ in lst])
            #[num actions x item]?
            trg_data, trg_mask = self.numeric_to_tensor([trg for _, trg in lst])
            # print("[src for src, _ in lst]",src_data.shape,src_mask.shape)
            # print("[trg for _, trg in lst]",trg_data.shape,trg_mask.shape)
            # input()
            self.batch_data[key] = (src_data, src_mask, trg_data, trg_mask)

        src_data, src_mask, trg_data, trg_mask = self.batch_data[key]
        nb_example = len(src_data[0])
        # print("nb_example",nb_example)
        if shuffle:
            # print("if")
            idx = np.random.permutation(nb_example)
        else:
            # print('else')
            idx = np.arange(nb_example)
        for start in range(0, nb_example, batch_size):
            idx_ = idx[start : start + batch_size]
            # print(idx_.shape)
            # input()
            src_mask_b = src_mask[:, idx_]
            trg_mask_b = trg_mask[:, idx_]
            src_len = int(src_mask_b.sum(dim=0).max().item())
            trg_len = int(trg_mask_b.sum(dim=0).max().item())
            src_data_b = src_data[:src_len, idx_].to(self.device)
            trg_data_b = trg_data[:, idx_].to(self.device)
            src_mask_b = src_mask_b[:src_len].to(self.device)
            #do not limit trg data by mask size
            trg_data_b = trg_data[:, idx_].to(self.device)
            trg_mask_b = trg_mask_b.to(self.device)

            print("src mask sizes and data sizes", src_data_b.size(), src_mask_b.size())
            print("trg mask sizes and data sizes", trg_data_b.size(), trg_mask_b.size())
            # print("target literal data", trg_data_b)

            yield (src_data_b, src_mask_b, trg_data_b, trg_mask_b)

class TransformerRegressor(TagTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_out = Linear(self.embed_dim, self.trg_vocab_size)

    def forward(self, src_batch, src_mask, trg_batch, trg_mask):
        src_mask = (src_mask == 0).transpose(0, 1)
        trg_mask = (trg_mask == 0).transpose(0, 1)
        # trg_seq_len, batch_size = trg_batch.size()

        enc_hs = self.encode(src_batch, src_mask)
        # output: [trg_seq_len, batch_size, vocab_siz]
        output = self.decode(enc_hs, src_mask, trg_batch, trg_mask)
        return output

    def decode(self, enc_hs, src_mask, trg_batch, trg_mask):
        trg_seq_len = 1 #not driven by data
        bos = torch.ones((1, trg_batch.size(1)), dtype=torch.long).to(DEVICE)

        #trg_seq_len = trg_batch.size(0)
        causal_mask = self.generate_square_subsequent_mask(trg_seq_len)
        # print("shape of causal mask", causal_mask.size())
        # print("shape of target mask", trg_mask.size())
        #assert(0)

        word_embed = self.embed_scale * self.trg_embed(bos)
        pos_embed = self.position_embed(bos)
        embed = self.dropout(word_embed + pos_embed)
        # print("shape of embed", bos.size(), word_embed.size(), pos_embed.size(), embed.size())

        causal_mask = self.generate_square_subsequent_mask(trg_seq_len)

        # for ind, tns in enumerate([embed, enc_hs, causal_mask, trg_mask, src_mask]):
        #     print("device check", ind, tns.get_device())

        dec_hs = self.decoder(
            embed,
            enc_hs,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=trg_mask,
            memory_key_padding_mask=src_mask,
        )
        return self.final_out(dec_hs)

    def get_loss(self, data, reduction=True):
        src, src_mask, trg, trg_mask = data
        # print("src",src.shape)
        # print("src_mask",src_mask.shape)
        # print("trg",trg.shape)
        # print("trg_mask",trg_mask.shape)
        # input()
        out = self.forward(src, src_mask, trg, trg_mask)
        loss = self.loss(out, trg, reduction=reduction)
        return loss

    def loss(self, predict, target, reduction=True):
        # print("raw shapes of pred", predict.size(), "targ", target.size())

        predict = predict.view(predict.size()[1:])
        target = target.transpose(0, 1)
        # print("LOSS: revised shapes of pred", predict.size(), "targ", target.size())

        reductionType = "none"
        if reduction:
            reductionType = "mean"

        loss = F.mse_loss(predict, target, reduction=reductionType)

        if not reduction: #report loss by batch item
            return loss.sum(1)

        return loss

