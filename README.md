## Description 
This is homework 1 of applied deep learning 2024 fall.
See NTU ADL 2024 Fall - HW1.pptx for details.

## Environment Set up
conda should be installed
1. `$git clone https://github.com/jnes0824/ADL_HW01.git`
2. `$cd ADL_HW01`
3. `$conda env create -f environment.yml`
4. `$conda activate ./env`(assume the env is created in the working dir)
5. check the dependency by running `python test.py`
6. `$git clone https://github.com/huggingface/transformers.git`
7. replace ./transformers/examples/pytorch/question-answering/run_swag_no_trainer.py with run_swag_no_trainer.py 
8. replace ./transformers/examples/pytorch/question-answering/run_qa_no_trainer.py with run_qa_no_trainer.py 
## Preprocessing Data
`$python ./data/preprocess_QA.py`
`$python ./data/preprocess_test.py`

## Training 
- training the multiple choice model
```
$accelerate launch ./transformers/examples/pytorch/question-answering/run_qa_no_trainer.py --model_name_or_path hfl/chinese-macbert-large --train_file ./data/QA_train_data.json --validation_file ./data/QA_validation_data.json --max_seq_length 512 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --num_train_epochs 2 --learning_rate 2e-5 --weight_decay 0.02 --lr_scheduler_type linear --with_tracking --report_to wandb --output_dir ./output_QA --checkpointing_steps epoch
```

- training the QA model
```
$accelerate launch ./transformers/examples/pytorch/multiple-choice/run_swag_no_trainer.py --model_name_or_path hfl/chinese-bert-wwm-ext --max_seq_length 512 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --num_train_epochs 3 --learning_rate 2e-5 --weight_decay 0.01 --lr_scheduler_type linear --checkpointing_steps 679 --train_file ./data/multiplechoice_train_data.json --validation_file ./data/multiplechoice_validation_data.json --with_tracking --report_to wandb --output_dir ./output_multiple_choice
```

## inference
`$python inference.py`

## submit to kaggle
`$kaggle competitions submit -c ntu-adl-2024-hw-1-chinese-extractive-qa -f submission.csv -m "Message"`