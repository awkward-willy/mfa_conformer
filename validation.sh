#!/bin/bash
# 設定參數
encoder_name="conformer_cat"  # conformer_cat | ecapa_tdnn_large | resnet34
embedding_dim=192
loss_name="amsoftmax"
num_classes=997
num_blocks=6
train_csv_path="data/CN_Celeb1+2_data.csv"
input_layer=conv2d2
pos_enc_layer_type=rel_pos   # no_pos|rel_pos
trial_path="data/MIRTest_parts.lst"

# 設定 save_dir
save_dir="validation_experiment/MIRTest/${input_layer}/${encoder_name}_${num_blocks}_${embedding_dim}_${loss_name}_CN_CN"
mkdir -p "$save_dir"

# # 將相關程式與資料複製到 save_dir 中（視需求而定）
# cp start.sh "$save_dir"
# cp main.py "$save_dir"
# cp -r module "$save_dir"
# cp -r wenet "$save_dir"
# cp -r scripts "$save_dir"
# cp -r loss "$save_dir"
# echo "save_dir: $save_dir"

# 設定 CUDA 裝置（若有需要的話）
export CUDA_VISIBLE_DEVICES=0

# 設定存放 ckpt 檔案的資料夾（請根據實際路徑修改）
ckpt_folder="/work/u4701865/mfa_conformer/experiment/conv2d2/conformer_cat_6_192_amsoftmax_CN_CN"

# 設定 log 檔案
log_file="${save_dir}/test_results.log"
if [ ! -f "$log_file" ]; then
    touch "$log_file"
fi
echo "開始測試，記錄存放於 ${log_file}" > "$log_file"

# 尋找資料夾中的所有 .ckpt 檔案，然後依序執行測試
find "$ckpt_folder" -type f -name "*.ckpt" | while read ckpt; do
    echo "============================" | tee -a "$log_file"
    echo "測試 checkpoint: $ckpt" | tee -a "$log_file"
    python3 main.py \
        --batch_size 200 \
        --num_workers 40 \
        --max_epochs 40 \
        --embedding_dim $embedding_dim \
        --save_dir "$save_dir" \
        --encoder_name "$encoder_name" \
        --train_csv_path "$train_csv_path" \
        --learning_rate 0.001 \
        --num_classes $num_classes \
        --trial_path "$trial_path" \
        --loss_name "$loss_name" \
        --num_blocks $num_blocks \
        --step_size 4 \
        --gamma 0.5 \
        --weight_decay 0.0000001 \
        --input_layer "$input_layer" \
        --pos_enc_layer_type "$pos_enc_layer_type" \
        --eval \
        --checkpoint_path "$ckpt" | tee -a "$log_file"
done
