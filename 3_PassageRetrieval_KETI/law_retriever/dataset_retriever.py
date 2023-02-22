
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_convert, default_collate
import json
import os
import bz2
import functools
import glob
import tqdm
import random
import multiprocessing
import copy
import csv

import logging


# from facebook research mdr github
def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


class QueryPassageFormatter(object):
    def __init__(
        self,
        tokenizer,
        max_q_len=50,
        max_q_sp_len=512,
        max_c_len=512,
        max_len=512,
        remove_question_mark=False,
        add_cls_token=False,
        add_sep_token=False,
        add_token_type_ids=True,
        cls_token_id=-1,
        sep_token_id=-1,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_q_sp_len = max_q_sp_len
        self.max_c_len = max_c_len
        self.max_len = max_len
        self.remove_question_mark = remove_question_mark

        self.add_cls_token = add_cls_token
        self.add_sep_token = add_sep_token
        self.add_token_type_ids = add_token_type_ids

        self.set_special_tokens(tokenizer, cls_token_id, sep_token_id)

    def set_special_tokens(self, tokenizer, cls_token_id, sep_token_id):
        if cls_token_id >= 0:
            self.cls_token_id = cls_token_id
        elif tokenizer.cls_token_id is not None:
            self.cls_token_id = tokenizer.cls_token_id
        elif len(tokenizer.additional_special_tokens_ids) > 1:
            self.cls_token_id = tokenizer.additional_special_tokens_ids[0]
        else:
            self.cls_token_id = tokenizer.pad_token_id

        if sep_token_id >= 0:
            self.sep_token_id = sep_token_id
        elif tokenizer.sep_token_id is not None:
            self.sep_token_id = tokenizer.sep_token_id
        elif len(tokenizer.additional_special_tokens_ids) > 1:
            self.sep_token_id = tokenizer.additional_special_tokens_ids[1]
        else:
            self.sep_token_id = tokenizer.pad_token_id

        self.pad_token_id = tokenizer.pad_token_id

        # special tokens for spans
        if len(tokenizer.additional_special_tokens_ids) > 2:
            self.sep_token_id_lv2 = tokenizer.additional_special_tokens_ids[2]
        else:
            self.sep_token_id_lv2 = self.sep_token_id

    def encode_para(self, para, max_len=512, add_cls_token=False, add_sep_token=False):
        return self.encode_pair(para["title"].strip(), para["text"].strip(), max_len=max_len, add_cls_token=add_cls_token, add_sep_token=add_sep_token)

    def encode_pair(self, text1, text2=None, max_len=512, add_cls_token=False, add_sep_token=False, convert2tensor=True):
        seq1 = self.tokenizer.encode_plus(
            text1,
            max_length=max_len,
            truncation=True,
            add_special_tokens=False)
        input_ids_s1 = seq1.input_ids
        attention_mask_s1 = seq1.attention_mask
        if add_cls_token:
            input_ids_s1 = [self.cls_token_id] + input_ids_s1
            attention_mask_s1 = [1] + attention_mask_s1

        input_ids_s2 = []
        attention_mask_s2 = []

        if text2 is not None:
            seq2 = self.tokenizer.encode_plus(
                text2,
                max_length=max_len,
                truncation=True,
                add_special_tokens=False)
            input_ids_s2 = seq2.input_ids
            attention_mask_s2 = seq2.attention_mask

        if add_sep_token:
            input_ids_s1 += [self.sep_token_id]
            attention_mask_s1 += [1]

            if text2 is not None:
                input_ids_s2 += [self.sep_token_id]
                attention_mask_s2 += [1]

        input_ids = input_ids_s1 + input_ids_s2
        attention_mask = attention_mask_s1 + attention_mask_s2

        return_val = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if 'token_type_ids' in seq1 and self.add_token_type_ids:
            return_val['token_type_ids'] = [0] * \
                len(input_ids_s1) + [1] * len(input_ids_s2)

        # truncate
        for k, v in return_val.items():
            if len(v) > max_len:
                return_val[k] = return_val[k][:max_len]

        # convert to tensor
        if convert2tensor:
            for k, v in return_val.items():
                return_val[k] = torch.LongTensor(return_val[k])

        return return_val

    def prepare_question(self, question):
        if self.remove_question_mark:
            if question.endswith("?"):
                question = question[:-1]
        return question

    def encode_question(self, question):
        question = self.prepare_question(question)

        return self.encode_pair(
            question,
            max_len=self.max_q_len,
            add_cls_token=self.add_cls_token,
            add_sep_token=self.add_sep_token)

    def encode_q_sp(self, question, passage):
        question = self.prepare_question(question)
        return self.encode_pair(
            question,
            passage["text"].strip(),
            max_len=self.max_q_sp_len,
            add_cls_token=self.add_cls_token,
            add_sep_token=self.add_sep_token)

    def encode_context(self, passage):
        return self.encode_para(
            passage,
            self.max_c_len,
            add_cls_token=self.add_cls_token,
            add_sep_token=self.add_sep_token)


class DatasetForRetriever(Dataset):
    def __init__(self,
                 qp_formatter,
                 data_path=None,
                 negative_sample="hard_negative_ctxs",
                 hard_positive=False,
                 extend_negative=True,
                 max_num_of_data=None
                 ):
        super().__init__()
        self.qp_formatter = qp_formatter
        self.negative_sample = negative_sample
        self.hard_positive = hard_positive

        if data_path is not None:
            logging.info(f"Loading data from {data_path}")
            self.data = json.load(open(data_path, "r"))
            if negative_sample != "negative_ctxs":
                for idx in range(len(self.data)):
                    if extend_negative:
                        self.data[idx]["negative_ctxs"].extend(self.data[idx][negative_sample])
                    else:
                        self.data[idx]["negative_ctxs"] = self.data[idx][negative_sample]
            self.data = [_ for _ in self.data if len(_["negative_ctxs"]) > 0]

            if max_num_of_data is not None:
                self.data = self.data[:max_num_of_data]

            logging.info(f"Total sample count {len(self.data)}")

    def __getitem__(self, index):
        sample = copy.deepcopy(self.data[index])
        question = sample['question']

        if self.hard_positive:
            positive_ctx = sample["positive_ctxs"][0]
        else:
            random.shuffle(sample["positive_ctxs"])
            positive_ctx = sample["positive_ctxs"][0]

        random.shuffle(sample["negative_ctxs"])
        negative_ctx = sample["negative_ctxs"][0]

        query = self.qp_formatter.encode_question(question)
        pos_ctx = self.qp_formatter.encode_context(positive_ctx)
        neg_ctx = self.qp_formatter.encode_context(negative_ctx)

        return {
            "query": query,
            "pos_ctx": pos_ctx,
            "neg_ctx": neg_ctx,
        }

    def __len__(self):
        return len(self.data)

    def get_collate_fn(self):
        return functools.partial(retriever_collate, pad_id=self.qp_formatter.pad_token_id)


class DatasetForRetrieverEvaluationOnNegativeSamples(Dataset):
    def __init__(self,
                 qp_formatter,
                 data_path=None
                 ):
        super().__init__()
        self.qp_formatter = qp_formatter
        self.pad_idx = qp_formatter.tokenizer.pad_token_id

        if data_path is not None:
            logging.info(f"Loading data from {data_path}")
            self.data = json.load(open(data_path, "r"))
            logging.info(f"Total sample count {len(self.data)}")
    
    def __getitem__(self, index):
        sample = copy.deepcopy(self.data[index])
        question = sample['question']

        positive_ctxs = sample["positive_ctxs"]
        negative_ctxs = sample["negative_ctxs"]

        query = [self.qp_formatter.encode_question(question)]
        example = query[0]
        query = {
            k: collate_tokens([s[k] for s in query], self.pad_idx if k=="input_ids" else 0) for k in example.keys()
        }

        pos_ctxs = [self.qp_formatter.encode_context(pc) for pc in positive_ctxs]
        neg_ctxs = [self.qp_formatter.encode_context(nc) for nc in negative_ctxs]

        pos_idx = random.randrange(len(neg_ctxs))

        ctxs = neg_ctxs[:pos_idx] + pos_ctxs + neg_ctxs[pos_idx:]
        example = ctxs[0]
        ctxs = {
            k: collate_tokens([s[k] for s in ctxs], self.pad_idx if k=="input_ids" else 0) for k in example.keys()
        }

        return {
            "query": query,
            "ctxs": ctxs,
            "pos_idx": pos_idx,
        }

    def __len__(self):
        return len(self.data)


class DatasetForRetrieverEvaluation(Dataset):
    def __init__(self,
                 qp_formatter,
                 data_path=None,
                 ):
        super().__init__()
        self.qp_formatter = qp_formatter

        if data_path is not None:
            logging.info(f"Loading data from {data_path}")
            self.data = json.load(open(data_path, "r"))
            logging.info(f"Total sample count {len(self.data)}")
    
    def __getitem__(self, index):
        sample = copy.deepcopy(self.data[index])
        question = sample['question']

        positive_ids = [int(pctx["passage_id"]) for pctx in sample["positive_ctxs"]]
        answers = sample["answers"]

        r_sample = self.qp_formatter.encode_question(question)

        r_sample["positive_ids"] = positive_ids
        r_sample["answers"] = answers


        r_sample["question"] = question
        r_sample["positive_ctxs"] = sample["positive_ctxs"]

        return r_sample

    def __len__(self):
        return len(self.data)

    def get_collate_fn(self):
        def collate_fn(samples, pad_id=0):
            if len(samples) == 0:
                return {}
            return {
                "input_ids": collate_tokens([s["input_ids"] for s in samples], pad_id),
                "attention_mask": collate_tokens([s["attention_mask"] for s in samples], 0),
                "c": [s["answers"] for s in samples],
                "positive_ids": [s["positive_ids"] for s in samples],
                "question": [s["question"] for s in samples],
                "positive_ctxs": [s["positive_ctxs"] for s in samples],
            }
        return functools.partial(collate_fn, pad_id=self.qp_formatter.pad_token_id)


# modified from facebook research mdr
def retriever_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    item = samples[0]

    return {
        k: {kk: collate_tokens([s[k][kk] for s in samples], pad_id if kk.endswith('input_ids') else 0) for kk in item[k].keys()} for k in item.keys()
    }


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def get_list(a, n, i):
    k, m = divmod(len(a), n)
    return a[i*k+min(i, m):(i+1)*k+min(i+1, m)]


class DatasetForWikipediaPassages100(Dataset):
    def __init__(self,
                 qp_formatter,
                 data_path=None,
                 shard_idx=0,
                 num_shards=1
                 ):
        super().__init__()
        self.qp_formatter = qp_formatter

        self.data = [{
                "id": int(row["id"]),
                "text": row["text"],
                "title": row["title"],
            } for row in csv.DictReader(open(data_path, mode='r'), delimiter="\t", quoting=csv.QUOTE_MINIMAL)]

        self.shard_idx = shard_idx
        if num_shards > 1:
            self.data = get_list(self.data, num_shards, shard_idx)
        
        logging.info("{} examples are loaded".format(len(self.data)))


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = copy.deepcopy(self.data[index])
        sample_ctx = self.qp_formatter.encode_context(sample)
        sample_ctx["id"] = sample["id"]

        return sample_ctx
    
    def get_collate_fn(self):
        def collate_fn(samples, pad_id=0):
            if len(samples) == 0:
                return {}
            return {
                "input_ids": collate_tokens([s["input_ids"] for s in samples], pad_id),
                "attention_mask": collate_tokens([s["attention_mask"] for s in samples], 0),
                "id": [s["id"] for s in samples]
            }
        return functools.partial(collate_fn, pad_id=self.qp_formatter.pad_token_id)




if __name__ == "__main__":
    from transformers import AutoTokenizer
    data_path = "law_data/retriever/law_qa_retriever_clean-dev.json"
    tokenizer_path = "KETI-AIR/ke-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    qp_formatter = QueryPassageFormatter(
        tokenizer
    )

    train_dataset = DatasetForRetriever(
        qp_formatter, 
        data_path,
        )

    for idx, item in zip(range(3), train_dataset):
        print(idx, item)
        print("query: ", tokenizer.decode(item["query"]["input_ids"]))
        print("pos_ctx: ", tokenizer.decode(item["pos_ctx"]["input_ids"]))
        print("neg_ctx: ", tokenizer.decode(item["neg_ctx"]["input_ids"]))




    # data_path = "law_data/law.tsv"
    # train_dataset = DatasetForWikipediaPassages100(
    #         qp_formatter, 
    #         data_path,
    #         shard_idx=1,
    #         num_shards=8
    #     )

    # print(len(train_dataset))
    
    # for idx, item in zip(range(3), train_dataset):
    #     print(idx, item)
    #     print("ctx: ", tokenizer.decode(item["input_ids"]))















