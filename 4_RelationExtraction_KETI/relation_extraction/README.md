# Relation extraction

## Install required packages
```bash
    pip install -r requirements.txt
```

## Create new datasets
```bash
    python download_and_write_datasets.py \
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
------------------------------
Total number of data: 40235
The number of data(train): 32188
The number of data(test): 8047
------------------------------
The number of duplicated examples between 'train' and 'test' splits:
0 examples
------------------------------
```

## Set accelerate config
```bash
accelerate config
In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU [4] MPS): 2
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
Do you want to use DeepSpeed? [yes/NO]: NO
Do you want to use FullyShardedDataParallel? [yes/NO]: NO
How many GPU(s) should be used for distributed training? [1]:2
Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: NO
```

## Training model

```bash
    # first dataset
    epochs=50
    learning_rate=0.0008
    scheduler_type=linear
    accelerate launch run_training.py \
        --data_path data/set_0 \
        --output_dir data/set_0 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --warmup_portion 0.02 \
        --learning_rate $learning_rate \
        --logging_steps 100 \
        --no_relation_weights 0.1 \
        --num_train_epochs $epochs \
        --with_tracking \
        --output_dir re_e${epochs}_${scheduler_type}_lr${learning_rate}_0
```

## Model evaluation
```bash
    python run_evaluation.py \
        --model_path result/re_e50_linear_lr0.0008_0 \
        --data_path data/set_0 \
        --number_of_print_examples 3 \
        --use_gpu
```
