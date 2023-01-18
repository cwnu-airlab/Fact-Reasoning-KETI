
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 run_training.py \
--tokenizer "KETI-AIR/ke-t5-small" \
--pre_trained_model "KETI-AIR/ke-t5-small" \
--config_path "KETI-AIR/ke-t5-small" \
--model_cls "T5MeanBiEncoderRetriever" \
--output_dir result/set_0 \
--extend_negative \
--epochs 160 \
--train_file data/set_0/train.json \
--dev_file data/set_0/test.json 


CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 run_training.py \
--tokenizer "KETI-AIR/ke-t5-small" \
--pre_trained_model "KETI-AIR/ke-t5-small" \
--config_path "KETI-AIR/ke-t5-small" \
--model_cls "T5MeanBiEncoderRetriever" \
--output_dir result/set_1 \
--extend_negative \
--epochs 160 \
--train_file data/set_1/train.json \
--dev_file data/set_1/test.json 


CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 run_training.py \
--tokenizer "KETI-AIR/ke-t5-small" \
--pre_trained_model "KETI-AIR/ke-t5-small" \
--config_path "KETI-AIR/ke-t5-small" \
--model_cls "T5MeanBiEncoderRetriever" \
--output_dir result/set_2 \
--extend_negative \
--epochs 160 \
--train_file data/set_2/train.json \
--dev_file data/set_2/test.json 


