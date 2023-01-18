import os
import logging
import argparse
import json
import random

from tqdm import tqdm
from datasets import load_dataset


logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Download KLUE RE datasets and randomly split it")
    parser.add_argument(
        "--target_dir",
        type=str,
        default="data/set_0",
        help="The path to target directory to check duplication",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    logger.info("Loading splits...")
    
    train_items = json.load(open(os.path.join(args.target_dir, "train.json"), "r"))
    test_items = json.load(open(os.path.join(args.target_dir, "test.json"), "r"))

    train_items_uniq = set([item["guid"] for item in train_items])

    cnt_dup = 0
    for item in tqdm(test_items, "check duplication..."):
        if item["guid"] in train_items_uniq:
            cnt_dup += 1
    
    print(
        "-"*30+"\n"
        + "Total number of data: {}\n".format(len(train_items) + len(test_items))
        + "The number of data(train): {}\n".format(len(train_items))
        + "The number of data(test): {}\n".format(len(test_items))
        +"-"*30+"\n"
        + "The number of duplicated examples between 'train' and 'test' splits:\n"
        + "{} examples\n".format(cnt_dup)
        +"-"*30+"\n"
    )
        

        


if __name__=="__main__":
    logging.basicConfig(level = logging.INFO)
    main()






