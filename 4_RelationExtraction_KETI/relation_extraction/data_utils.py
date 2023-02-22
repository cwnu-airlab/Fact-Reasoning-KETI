import re
import os
import json
import logging
import functools

import torch
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)

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

class KlueREDataset(object):
    def __init__(self, tokenizer, data_dir="data/set_0", split="train", max_num_of_data=None) -> None:
        self.tokenizer = tokenizer
        self.split = split
        file_path = os.path.join(data_dir, "train.json") if split=="train" else os.path.join(data_dir, "test.json")
        logger.info("Loading datasets({}) - [{}]".format(split, file_path))
        self.data = json.load(open(file_path, "r"))
        if max_num_of_data is not None:
            self.data = self.data[:max_num_of_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        item = self.data[index]
        sample = self.re_preproc_for_classification_with_idx(item)
        tk_idxs = self.tokenize_re_with_tk_idx(sample)

        return {
            "id": sample["id"],
            "input_ids": tk_idxs["input_ids"],
            "attention_mask": tk_idxs["attention_mask"],
            "labels": torch.tensor(sample["targets"]),
            "subject_token_idx": tk_idxs["subject_token_idx"],
            "object_token_idx": tk_idxs["object_token_idx"]
        }

    def re_preproc_for_classification_with_idx(
        self,
        x,
        benchmark_name="klue_re",
        label_names=None,
        no_label_idx=0,
        with_feature_key=False,
        sep=' '):
        # mark span using start index of the entity
        def _mark_span(text, span_str, span_idx, mark):
            pattern_tmpl = r'^((?:[\S\s]){N})(W)'
            pattern_tmpl = pattern_tmpl.replace('N', str(span_idx))
            pattern = pattern_tmpl.replace('W', span_str)
            return re.sub(pattern, r'\1{0}\2{0}'.format(mark), text)

        def _mark_span_sub(text, start_idx, end_idx, mark):
            end_idx += 2
            text = text[:start_idx] + mark + text[start_idx:]
            text = text[:end_idx] + mark + text[end_idx:]
            return text

        # '*' for subejct entity '#' for object entity
        text = x["sentence"]

        text = _mark_span_sub(text,
                              x['subject_entity']['start_idx'],
                              x['subject_entity']['end_idx'],
                              '*')

        sbj_st, sbj_end = x['subject_entity']['start_idx'], x['subject_entity']['end_idx']
        obj_st, obj_end = x['object_entity']['start_idx'], x['object_entity']['end_idx']
        sbj_end += 3
        obj_end += 3
        if sbj_st < obj_st:
            obj_st += 2
            obj_end += 2
        else:
            sbj_st += 2
            sbj_end += 2

        # Compensate for 2 added "words" added in previous step
        span2st = x['object_entity']['start_idx'] + 2 * (1 if x['subject_entity']['start_idx'] < x['object_entity']['start_idx'] else 0)
        span2et = x['object_entity']['end_idx'] + 2 * (1 if x['subject_entity']['end_idx'] < x['object_entity']['end_idx'] else 0)
        text = _mark_span_sub(text, span2st, span2et, '#')

        strs_to_join = []
        if with_feature_key:
            strs_to_join.append('{}:'.format('text'))
        strs_to_join.append(text)

        ex = {}

        if label_names is not None:
            # put the name of benchmark if the model is generative
            strs_to_join.insert(0, benchmark_name)
            ex['targets'] = label_names[x['label']] if x['label'] >= 0 else '<unk>'
        else:
            ex['targets'] = x['label'] if x['label'] >= 0 else no_label_idx

        offset = len(sep.join(strs_to_join[:-1] +['']))
        sbj_st+=offset
        sbj_end+=offset
        obj_st+=offset
        obj_end+=offset

        ex['subject_entity'] = {
            "type": x['subject_entity']['type'],
            "start_idx": sbj_st,
            "end_idx": sbj_end,
            "word": x['subject_entity']['word'],
        }
        ex['object_entity'] = {
            "type": x['object_entity']['type'],
            "start_idx": obj_st,
            "end_idx": obj_end,
            "word": x['object_entity']['word'],
        }

        joined = sep.join(strs_to_join)
        ex['inputs'] = joined
        ex['id'] = x['guid']

        return ex

    def tokenize_re_with_tk_idx(self, x, input_key='inputs'):
        ret = {}

        inputs = x[input_key]
        ret[f'{input_key}_pretokenized'] = inputs
        input_hf = self.tokenizer(inputs, padding=True, truncation='longest_first', return_tensors='pt')
        input_ids = input_hf.input_ids
        attention_mask = input_hf.attention_mask

        subject_entity = x['subject_entity']
        object_entity = x['object_entity']

        subject_tk_idx = [
            input_hf.char_to_token(x) for x in range(
                subject_entity['start_idx'],
                subject_entity['end_idx']
                )
            ]
        subject_tk_idx = [x for x in subject_tk_idx if x is not None]
        subject_tk_idx = sorted(set(subject_tk_idx))
        subject_start = subject_tk_idx[0]
        subject_end = subject_tk_idx[-1]

        object_tk_idx = [
            input_hf.char_to_token(x) for x in range(
                object_entity['start_idx'],
                object_entity['end_idx']
                )
            ]
        object_tk_idx = [x for x in object_tk_idx if x is not None]
        object_tk_idx = sorted(set(object_tk_idx))
        object_start = object_tk_idx[0]
        object_end = object_tk_idx[-1]

        subject_token_idx = torch.zeros_like(input_ids)
        object_token_idx = torch.zeros_like(input_ids)
        subject_token_idx[0, subject_start:subject_end] = 1
        object_token_idx[0, object_start:object_end] = 1

        ret['subject_token_idx'] = subject_token_idx
        ret['object_token_idx'] = object_token_idx
        ret['input_ids'] = input_ids
        ret['attention_mask'] = attention_mask

        return ret

    def get_collate_fn(self):
        def collate_fn(batch, pad_id):
            if len(batch) == 0:
                return None

            collated_batch = {
                "input_ids": collate_tokens([ex["input_ids"] for ex in batch], pad_id),
                "attention_mask": collate_tokens([ex["attention_mask"] for ex in batch], 0),
                "labels" : default_collate([ex["labels"] for ex in batch]),
                "subject_token_idx": collate_tokens([ex["subject_token_idx"] for ex in batch], 0),
                "object_token_idx": collate_tokens([ex["object_token_idx"] for ex in batch], 0),
            }

            return collated_batch

        return functools.partial(collate_fn, pad_id=self.tokenizer.pad_token_id)


if __name__=="__main__":
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained("KETI-AIR/ke-t5-base")
    data_dir = "data/set_0"
    dataset = KlueREDataset(tokenizer, data_dir, split="train")

    for item in dataset:
        print(tokenizer.decode(item["input_ids"][0]))
        print(tokenizer.decode(torch.einsum("i,i->i", item["input_ids"][0], item["subject_token_idx"][0])))
        print(tokenizer.decode(torch.einsum("i,i->i", item["input_ids"][0], item["object_token_idx"][0])))
        exit()


