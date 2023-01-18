

import argparse
import logging

import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

from dataset_retriever import DatasetForRetrieverEvaluationOnNegativeSamples, QueryPassageFormatter
from modeling_retriever import (
    T5SimpleMomentumRetriever,
    T5MeanMomentumRetriever,
    T5SimpleBiEncoderRetriever,
    T5MeanBiEncoderRetriever
)


logger = logging.getLogger(__name__)


MODEL_CLS = {
    "T5SimpleMomentumRetriever": {
        "model_cls": T5SimpleMomentumRetriever,
    },
    "T5MeanMomentumRetriever": {
        "model_cls": T5MeanMomentumRetriever,
    },
    "T5SimpleBiEncoderRetriever": {
        "model_cls": T5SimpleBiEncoderRetriever,
    },
    "T5MeanBiEncoderRetriever": {
        "model_cls": T5MeanBiEncoderRetriever,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--model_path",
                        default="./result/T5MeanBiEncoderRetriever_KETI-AIR_ke-t5-small_ext_e160/hf", type=str)
    parser.add_argument("--model_cls", default="T5MeanBiEncoderRetriever", 
                        choices=["T5SimpleMomentumRetriever", 
                                "T5MeanMomentumRetriever", 
                                "T5SimpleBiEncoderRetriever", 
                                "T5MeanBiEncoderRetriever"],
                        type=str, help="model class")
    
    # data file
    parser.add_argument("--file_path", type=str,
                        default="./law_data/retriever/law_qa_retriever_clean-dev.json")

    # encoding
    parser.add_argument("--max_q_len", default=70, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--max_q_sp_len", default=350, type=int)
    parser.add_argument("--max_c_len", default=350, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    # dataset setting
    parser.add_argument("--remove_question_mark", action='store_true',
                        help="remove_question_mark")
    parser.add_argument("--wo_cls_token", action='store_false',
                        help="wo_cls_token")
    parser.add_argument("--wo_sep_token", action='store_false',
                        help="wo_sep_token")
    parser.add_argument("--add_token_type_ids", action='store_false',
                        help="add_token_type_ids")
    parser.add_argument("--cls_token_id", type=int, default=-1,
                        help="cls_token_id")
    parser.add_argument("--sep_token_id", type=int, default=-1,
                        help="sep_token_id")
    parser.add_argument("--negative_sample", default="hard_negative_ctxs",
                        type=str, choices=[
                            "negative_ctxs", "hard_negative_ctxs"],
                        help="negative sample for training")
    parser.add_argument("--extend_negative", action='store_true',
                        help="extend_negative")

    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Whether to use gpu",
    )


    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    device = torch.device('cpu')
    if args.use_gpu:
        device = torch.device('cuda')

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # create query formatter
    qp_formatter = QueryPassageFormatter(
        tokenizer,
        max_q_len=args.max_q_len,
        max_q_sp_len=args.max_q_sp_len,
        max_c_len=args.max_c_len,
        remove_question_mark=args.remove_question_mark,
        add_cls_token=args.wo_cls_token,
        add_sep_token=args.wo_sep_token,
        add_token_type_ids=args.add_token_type_ids,
        cls_token_id=args.cls_token_id,
        sep_token_id=args.sep_token_id,
    )

    # get model
    model_cls_cfg = MODEL_CLS[args.model_cls]
    model_cls = model_cls_cfg["model_cls"]
    model = model_cls.from_pretrained(args.model_path)
    model = model.to(device)


    test_dataset = DatasetForRetrieverEvaluationOnNegativeSamples(
            qp_formatter, 
            args.file_path,
        )
    
    model.eval()

    labels = []
    prediction = []

    with torch.no_grad():
        for item in tqdm(test_dataset):
            pos_idx = item["pos_idx"]
            item["query"]
            qeury = model.encode_query({
                k: v.to(device) for k, v in item["query"].items()
            })
            context = model.encode_context({
                k: v.to(device) for k, v in item["ctxs"].items()
            })

            scores = torch.mm(qeury.squeeze(1), context.squeeze(1).t())
            index = torch.argmax(scores, dim=-1).cpu().item()
            labels.append(pos_idx)
            prediction.append(index)

    number_of_examples = len(labels)
    labels = np.array(labels)
    prediction = np.array(prediction)
    eq_num = np.sum((labels == prediction))
    em_score = eq_num/number_of_examples*100
    print("Number of samples: {}\nNumber of correct predictions: {}\nExact match score: {:.3f}".format(
            number_of_examples,
            eq_num,
            em_score
        ))

    


if __name__=="__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    main()
