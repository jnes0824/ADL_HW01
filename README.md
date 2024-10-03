```
/home/tingjung/MS2-S1/ADL/transformers/examples/pytorch/question-answering/run_qa_no_trainer.py --model_name_or_path hfl/chinese-macbert-large --train_file ./data/QA_train_data.json --validation_file ./data/QA_validation_data.json --max_seq_length 512 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --num_train_epochs 2 --learning_rate 2e-5 --weight_decay 0.02 --lr_scheduler_type linear --with_tracking --report_to wandb --output_dir ./output_QA8 --checkpointing_steps epoch
```


```
/home/tingjung/MS2-S1/ADL/transformers/examples/pytorch/multiple-choice/run_swag_no_trainer.py --model_name_or_path hfl/chinese-bert-wwm-ext --max_seq_length 512 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --num_train_epochs 3 --learning_rate 2e-5 --weight_decay 0.01 --lr_scheduler_type linear --checkpointing_steps 679 --train_file ./data/multiplechoice_train_data.json --validation_file ./data/multiplechoice_validation_data.json --with_tracking --report_to wandb --output_dir ./output_multiple_choice3
```