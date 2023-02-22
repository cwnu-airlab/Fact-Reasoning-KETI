
epochs=50
learning_rate=0.0008
scheduler_type=linear

accelerate launch run_training.py \
--data_path data/set_0 \
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

accelerate launch run_training.py \
--data_path data/set_1 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--warmup_portion 0.02 \
--learning_rate $learning_rate \
--logging_steps 100 \
--no_relation_weights 0.1 \
--num_train_epochs $epochs \
--with_tracking \
--output_dir re_e${epochs}_${scheduler_type}_lr${learning_rate}_1

accelerate launch run_training.py \
--data_path data/set_2 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--warmup_portion 0.02 \
--learning_rate $learning_rate \
--logging_steps 100 \
--no_relation_weights 0.1 \
--num_train_epochs $epochs \
--with_tracking \
--output_dir re_e${epochs}_${scheduler_type}_lr${learning_rate}_2

