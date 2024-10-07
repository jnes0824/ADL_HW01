我是將multiple choice 問題用not pre-trained 在訓練一次，tokenizer用bert-base-uncased ，以下是training hyperparameters
```
python ./from_scratch.py \
  --model_type bert \
  --tokenizer_name bert-base-uncased \
  --max_seq_length 512 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --lr_scheduler_type linear \
  --train_file ./data/multiplechoice_train_data.json \
  --validation_file ./data/multiplechoice_validation_data.json \
  --with_tracking \
  --report_to wandb \
  --output_dir ./from_scratch \
  --checkpointing_steps epoch
```
在validation上的performance，最後只有0.52跟pretrain model差很多
![W&B Chart 2024_10_7 下午4_14_21](https://hackmd.io/_uploads/Sk5_OzbJyx.png)
loss也降不太下來
![W&B Chart 2024_10_7 下午4_15_43](https://hackmd.io/_uploads/S1fWiMW1kx.png)