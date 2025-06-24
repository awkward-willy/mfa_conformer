#!/bin/bash
# 設定參數
# enrollment 資料夾中每個子資料夾代表一個語者，內含該語者的多支音檔
enroll_dir="/work/u4701865/downloads/MIRTest/parts"  
# 模型 checkpoint 路徑
checkpoint_path="/work/u4701865/mfa_conformer/experiment/conv2d2/conformer_cat_6_192_amsoftmax_CN1+2_CN/epoch=39_cosine_eer=1.06.ckpt"  
# enrollment 階段產生的 embedding 輸出檔案
output="CN_1+2_enroll_embeddings.npy"  
# 待辨識音檔列表檔案，每行一個 wav 檔路徑
recog_lst="/work/u4701865/mfa_conformer/data/recog.lst"  

# 模型參數（依照 main.py 與 validation.sh 中設定）
encoder_name="conformer_cat"
embedding_dim=192
num_blocks=6
loss_name="amsoftmax"
# CN1
# num_classes=997
# CN 1+2
num_classes=2993
train_csv_path="data/CN_Celeb1+2_data.csv"
input_layer="conv2d2"
pos_enc_layer_type="rel_pos"

echo "=== Enrollment 階段 ==="
python3 ASR.py --mode enroll \
    --enroll_dir "$enroll_dir" \
    --checkpoint_path "$checkpoint_path" \
    --output "$output" \
    --encoder_name "$encoder_name" \
    --embedding_dim $embedding_dim \
    --num_blocks $num_blocks \
    --loss_name "$loss_name" \
    --num_classes $num_classes \
    --train_csv_path "$train_csv_path" \
    --input_layer "$input_layer" \
    --pos_enc_layer_type "$pos_enc_layer_type"

echo "=== Recognition 階段 ==="
python3 ASR.py --mode recognize \
    --recog_lst "$recog_lst" \
    --checkpoint_path "$checkpoint_path" \
    --enroll_embeddings "$output" \
    --encoder_name "$encoder_name" \
    --embedding_dim $embedding_dim \
    --num_blocks $num_blocks \
    --loss_name "$loss_name" \
    --num_classes $num_classes \
    --train_csv_path "$train_csv_path" \
    --input_layer "$input_layer" \
    --pos_enc_layer_type "$pos_enc_layer_type"
