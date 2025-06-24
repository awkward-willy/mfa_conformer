encoder_name="conformer_cat" # conformer_cat | ecapa_tdnn_large | resnet34
embedding_dim=192
loss_name="amsoftmax"

dataset="vox"
# num_classes=7205
# num_classes=997
num_classes=2993
num_blocks=6
# train_csv_path="data/train_vox1_vox2.csv"
# train_csv_path="data/vox2_train_dev.csv"
# train_csv_path="data/CN_Celeb_data.csv"
train_csv_path="data/CN_Celeb1+2_data.csv"
input_layer=conv2d2
pos_enc_layer_type=rel_pos # no_pos| rel_pos 
# save_dir=experiment/${input_layer}/${encoder_name}_${num_blocks}_${embedding_dim}_${loss_name}_CN1+2_CN
save_dir=validation_experiment/${input_layer}/${encoder_name}_${num_blocks}_${embedding_dim}_${loss_name}_CN1+2_CN
# trial_path=data/trial.lst
# trial_path=data/vox1_test.txt
# trial_path=data/CN_Celeb_trials.lst
trial_path=data/lnflux_test.txt

mkdir -p $save_dir
# cp start.sh $save_dir
# cp main.py $save_dir
# cp -r module $save_dir
# cp -r wenet $save_dir
# cp -r scripts $save_dir
# cp -r loss $save_dir
echo save_dir: $save_dir

export CUDA_VISIBLE_DEVICES=0
python3 main.py \
        --batch_size 200 \
        --num_workers 40 \
        --max_epochs 40 \
        --embedding_dim $embedding_dim \
        --save_dir $save_dir \
        --encoder_name $encoder_name \
        --train_csv_path $train_csv_path \
        --learning_rate 0.001 \
        --encoder_name ${encoder_name} \
        --num_classes $num_classes \
        --trial_path $trial_path \
        --loss_name $loss_name \
        --num_blocks $num_blocks \
        --step_size 4 \
        --gamma 0.5 \
        --weight_decay 0.0000001 \
        --input_layer $input_layer \
        --pos_enc_layer_type $pos_enc_layer_type \
	--eval \
	--checkpoint_path /work/u4701865/mfa_conformer/experiment/conv2d2/conformer_cat_6_192_amsoftmax_CN1+2_CN/epoch=39_cosine_eer=1.06.ckpt 
