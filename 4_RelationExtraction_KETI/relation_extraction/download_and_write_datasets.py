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
        "--output_root_dir",
        type=str,
        default="data",
        help="The path to output root directory.",
    )
    parser.add_argument(
        "--number_of_split",
        type=int,
        default=3,
        help="The number of splits",
    )
    parser.add_argument(
        "--test_proportion",
        type=int,
        default=20,
        help="The percent of test split",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    logger.info("Download KLUE RE datasets...")
    data_train = load_dataset("klue", "re")["train"]
    data_validation = load_dataset("klue", "re")["validation"]

    item_list = []
    for item in data_train:
        item_list.append(item)
    for item in data_validation:
        item_list.append(item)
    
    number_of_dataset = len(item_list)
    nv = int(number_of_dataset*args.test_proportion/100)
    nt = number_of_dataset - nv

    logger.info(
        "-"*30+"\n"
        +"Total number of examples: {}\n".format(number_of_dataset)
        +"The number of examples for 'train' split: {}\n".format(nt)
        +"The number of examples for 'test' split: {}\n".format(nv)
        + "-"*30+"\n"
        )

    os.makedirs(args.output_root_dir, exist_ok=True)

    for split_idx in tqdm(range(args.number_of_split), desc="split dataset..."):
        random.shuffle(item_list)

        item4train, item4test = item_list[:nt], item_list[nt:]

        output_dir = os.path.join(args.output_root_dir, "set_{}".format(split_idx))
        os.makedirs(output_dir, exist_ok=True)

        train_data_path = os.path.join(output_dir, "train.json")
        json.dump(item4train, open(train_data_path, "w"), indent=4)

        test_data_path = os.path.join(output_dir, "test.json")
        json.dump(item4test, open(test_data_path, "w"), indent=4)
        


if __name__=="__main__":
    logging.basicConfig(level = logging.INFO)
    main()






