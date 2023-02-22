
import argparse

from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from data_utils import KlueREDataset
from metrics import f1_score_micro_sample_weight_dict
from models import T5EncoderForSequenceClassificationFirstSubmeanObjmean
from meta_data import _KLUE_RE_RELATIONS


def parse_args():
    parser = argparse.ArgumentParser()

    # default settings
    parser.add_argument("--model_path",
                        default="re_e50_linear_lr0.0008", type=str)
    parser.add_argument("--data_path",
                        default="data/set_0", type=str)
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Whether to use gpu",
    )
    parser.add_argument(
        "--number_of_print_examples",
        type=int,
        default=3,
        help="The number of examples to print",
    )
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    device = torch.device('cpu')
    if args.use_gpu:
        device = torch.device('cuda')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = T5EncoderForSequenceClassificationFirstSubmeanObjmean.from_pretrained(
            args.model_path,
        )
    model = model.to(device)

    eval_dataset = KlueREDataset(tokenizer, data_dir=args.data_path, split="test")
    collate_fn = eval_dataset.get_collate_fn()
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size
    )


    model.eval()

    indice_all, labels_all = [], []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="predict relation..."):
            outputs = model(
                input_ids = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                subject_token_idx = batch["subject_token_idx"].to(device),
                object_token_idx = batch["object_token_idx"].to(device),
            )
            logits = outputs.logits
            indice = torch.argmax(logits, dim=-1)
            labels = batch["labels"]

            indice_all.append(indice.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    indice_all = np.concatenate(indice_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    
    predictions = [model.config.id2label[idx] for idx in indice_all]

    f1_score_wo_no_relation = f1_score_micro_sample_weight_dict(
        labels_all,
        indice_all, 
        ex_num=_KLUE_RE_RELATIONS.index("no_relation"),
    )

    f1_score= f1_score_micro_sample_weight_dict(
        labels_all, 
        indice_all, 
    )

    print("f1_score: {}\nf1_score_wo_no_relation: {}\n".format(f1_score, f1_score_wo_no_relation))

    for idx, predicted_label, item in zip(range(args.number_of_print_examples), predictions, eval_dataset):
        input_ids = item["input_ids"]
        labels = item["labels"].numpy()
        
        text = tokenizer.decode(input_ids[0])
        subject_span = tokenizer.decode(
            torch.einsum("i,i->i", item["input_ids"][0], item["subject_token_idx"][0]),
            skip_special_tokens=True,
        )
        object_span = tokenizer.decode(
            torch.einsum("i,i->i", item["input_ids"][0], item["object_token_idx"][0]),
            skip_special_tokens=True,
        )
        labels_text = model.config.id2label[int(labels)]
        print("input text: {}\nsubject: {}\nobject: {}\norigin_label: {}\npredicted label: {}\n".format(
            text,
            subject_span,
            object_span,
            labels_text,
            predicted_label,
        ))


if __name__=="__main__":
    main()
