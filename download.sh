#!/bin/bash

# 下載文件的 ID，從 Google Drive 分享鏈接提取
FILE_ID="1wQBPD4Jw4WiqZeQCXFkUuyJ7r0-O8RxR"

# 下載後的文件名稱
FILE_NAME="model.zip"

# 檢查是否安裝了 gdown
if ! command -v gdown &> /dev/null; then
    echo "gdown 未安裝，正在安裝..."
    pip install gdown
fi

# 使用 gdown 從 Google Drive 下載文件
echo "正在下載文件..."
gdown https://drive.google.com/uc?id=$FILE_ID -O $FILE_NAME

# 檢查文件是否成功下載
if [ $? -ne 0 ]; then
    echo "文件下載失敗"
    exit 1
fi

# 解壓縮文件
echo "正在解壓縮 $FILE_NAME..."
unzip $FILE_NAME 

# 檢查解壓縮是否成功
if [ $? -ne 0 ]; then
    echo "解壓縮失敗"
    exit 1
fi

echo "文件下載並解壓縮成功"

# 刪除 ZIP 文件
rm $FILE_NAME
