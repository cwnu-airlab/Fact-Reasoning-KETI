# Law retriever

## Install required packages
```bash
    pip install -r requirements.txt
```

## Create new datasets
```bash
    python split_datasets.py \
        --original_data_root data/original \
        --output_root_dir data \
        --number_of_split 3 \
        --test_proportion 20
```

## Check duplication
```bash
    # first dataset
    python check_duplication_btw_splits.py \
        --target_dir data/set_0

    # second dataset
    python check_duplication_btw_splits.py \
        --target_dir data/set_1
    
    # third dataset
    python check_duplication_btw_splits.py \
        --target_dir data/set_2
```

```bash

python check_duplication_btw_splits.py \
        --target_dir data/set_0


# shell
INFO:__main__:Loading splits...
check duplication...: 100%|████████████████████████████████████████████████████████████████████| 1614/1614 [00:00<00:00, 1237814.35it/s]
------------------------------
Total number of data: 8074
The number of data(train): 6460
The number of data(test): 1614
------------------------------
The number of duplicated examples between 'train' and 'test' splits:
64 examples
------------------------------

remove duplicated examples from test dataset...: 100%|███████████████████████████████████| 1614/1614 [00:00<00:00, 1827647.59it/s]
INFO:__main__:Write 1550 examples saved in data/set_0/test.json



python check_duplication_btw_splits.py \
        --target_dir data/set_0


# shell
INFO:__main__:Loading splits...
check duplication...: 100%|████████████████████████████████████████████████████████████████████| 1550/1550 [00:00<00:00, 1253117.04it/s]
------------------------------
Total number of data: 8010
The number of data(train): 6460
The number of data(test): 1550
------------------------------
The number of duplicated examples between 'train' and 'test' splits:
0 examples
------------------------------
```


## Training model

```bash
    # first dataset
    CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 run_training.py \
        --tokenizer "KETI-AIR/ke-t5-small" \
        --pre_trained_model "KETI-AIR/ke-t5-small" \
        --config_path "KETI-AIR/ke-t5-small" \
        --model_cls "T5MeanBiEncoderRetriever" \
        --output_dir result/set_0 \
        --extend_negative \
        --epochs 160 \
        --train_file data/set_0/train.json \
        --dev_file data/set_0/train.json 
```

## Model evaluation
```bash
    python run_evaluation.py \
        --model_path result/set_0/hf \
        --model_cls "T5MeanBiEncoderRetriever" \
        --file_path data/set_0/test.json \
        --use_gpu
```
