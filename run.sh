#!/bin/bash

# 檢查是否提供了足夠的參數
if [ "$#" -ne 3 ]; then
    echo "用法: ./run.sh <context.json 路徑> <test.json 路徑> <prediction.csv 路徑>"
    exit 1
fi

# 取得參數
CONTEXT_FILE=$1
TEST_FILE=$2
PREDICTION_FILE=$3

# 運行 Python 程序進行推理
echo "正在運行推理模型..."
python ./inference.py $CONTEXT_FILE $TEST_FILE $PREDICTION_FILE

# 檢查推理是否成功
if [ $? -eq 0 ]; then
    echo "推理完成，預測結果保存在 $PREDICTION_FILE"
else
    echo "推理失敗"
    exit 1
fi
