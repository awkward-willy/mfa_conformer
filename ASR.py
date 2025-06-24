#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torchaudio

# 假設原 repo 中的前處理模組與模型架構皆可匯入
from module.feature import Mel_Spectrogram
from main import Task  # 從 main.py 載入模型定義

def load_audio(wav_path, target_sr=16000):
    # 使用 torchaudio 載入，假設模型的取樣率為 16kHz
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform

def compute_embedding(model, waveform, device):
    model.eval()
    with torch.no_grad():
        x = waveform.to(device)
        # 調整輸入維度：mel_trans 預期輸入 shape 為 (batch, time)
        if len(x.shape) == 2:
            # 若有多通道，先取平均；若單通道則 squeeze 掉 channel 維度
            if x.shape[0] > 1:
                x = torch.mean(x, dim=0)
            else:
                x = x.squeeze(0)
            # 加上 batch 維度
            x = x.unsqueeze(0)
        elif len(x.shape) == 1:
            # 若僅為 1D，補上 batch 維度
            x = x.unsqueeze(0)
        embedding = model(x)
    return embedding.cpu().numpy().squeeze()

def enroll(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型並載入 ckpt 檔
    model = Task(
        encoder_name=args.encoder_name,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        loss_name=args.loss_name,
        num_classes=args.num_classes,
        train_csv_path=args.train_csv_path,
        input_layer=args.input_layer,
        pos_enc_layer_type=args.pos_enc_layer_type,
        trial_path="/work/u4701865/mfa_conformer/data/trial.lst"  # enroll 階段不需要 trial.lst
    )
    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    enrollment_embeddings = {}
    # 對 enrollment 資料夾中每個子目錄（代表一個語者）處理
    for speaker in os.listdir(args.enroll_dir):
        speaker_dir = os.path.join(args.enroll_dir, speaker)
        if os.path.isdir(speaker_dir):
            embeddings = []
            # 遞迴讀取所有 .wav 檔
            for root, _, files in os.walk(speaker_dir):
                for f in files:
                    if f.lower().endswith('.wav'):
                        wav_path = os.path.join(root, f)
                        waveform = load_audio(wav_path)
                        emb = compute_embedding(model, waveform, device)
                        embeddings.append(emb)
            if embeddings:
                avg_emb = np.mean(embeddings, axis=0)
                enrollment_embeddings[speaker] = avg_emb
                print(f"Speaker {speaker}: {len(embeddings)} files, averaged embedding computed.")
            else:
                print(f"Speaker {speaker}: No wav files found.")
    # 儲存成 npy 檔（後續 recognition 階段讀取用）
    np.save(args.output, enrollment_embeddings)
    print("Enrollment embeddings saved to", args.output)

def recognize(args):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型並載入 ckpt
    model = Task(
        encoder_name=args.encoder_name,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        loss_name=args.loss_name,
        num_classes=args.num_classes,
        train_csv_path=args.train_csv_path,
        input_layer=args.input_layer,
        pos_enc_layer_type=args.pos_enc_layer_type,
        trial_path="/work/u4701865/mfa_conformer/data/trial.lst"  # 此參數不會用到
    )
    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    
    # 載入 enrollment 階段的 embedding 檔（dict: speaker -> embedding）
    enrollment_embeddings = np.load(args.enroll_embeddings, allow_pickle=True).item()

    # 讀取 recog.lst 中每一行的音檔路徑
    with open(args.recog_lst, 'r') as f:
        recog_files = [line.strip() for line in f if line.strip()]

    predictions = []
    ground_truths = []

    for wav_path in recog_files:
        waveform = load_audio(wav_path)
        rec_emb = compute_embedding(model, waveform, device)
        best_speaker = None
        best_score = -1
        # 以 cosine similarity 作比對
        for speaker, emb in enrollment_embeddings.items():
            score = np.dot(rec_emb, emb) / (np.linalg.norm(rec_emb) * np.linalg.norm(emb) + 1e-8)
            if score > best_score:
                best_score = score
                best_speaker = speaker

        # 根據檔名取得 ground truth
        base = os.path.basename(wav_path)
        gt = base.split('-')[0].split('.')[0]
        ground_truths.append(gt)
        predictions.append(best_speaker)

        print(f"File: {wav_path}")
        print(f"  Ground Truth: {gt}")
        print(f"  Recognized Speaker: {best_speaker}  (cosine similarity: {best_score:.4f})")

    # 計算 Accuracy
    correct = sum([1 for p, g in zip(predictions, ground_truths) if p == g])
    accuracy = correct / len(ground_truths)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # 混淆矩陣
    labels = sorted(list(set(ground_truths)))
    cm = confusion_matrix(ground_truths, predictions, labels=labels)
    print("\nClassification Report:")
    print(classification_report(ground_truths, predictions, labels=labels))

    # 繪製混淆矩陣，並在每個格子中加入數字
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    # 在每個格子內加入數字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as confusion_matrix.png")
    # plt.show()  # 若不需要顯示圖像則可註解掉


def main():
    parser = argparse.ArgumentParser(description="ASR for Speaker Recognition (Enroll & Recognize)")
    parser.add_argument('--mode', type=str, choices=['enroll', 'recognize'], required=True,
                        help="選擇模式：enroll 或 recognize")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="模型 ckpt 路徑")
    # enroll 模式參數
    parser.add_argument('--enroll_dir', type=str, help="enroll 資料夾路徑，每個子目錄為一個語者")
    parser.add_argument('--output', type=str, default='enroll_embeddings.npy', help="存放 enrollment embeddings 的檔案")
    # recognize 模式參數
    parser.add_argument('--recog_lst', type=str, help="待辨識音檔列表，每行一個 wav 路徑")
    parser.add_argument('--enroll_embeddings', type=str, help="預先產生的 enrollment embeddings 檔案")
    # 模型參數（參照 main.py 預設參數）
    parser.add_argument('--encoder_name', type=str, default='conformer_cat')
    parser.add_argument('--embedding_dim', type=int, default=192)
    parser.add_argument('--num_blocks', type=int, default=6)
    parser.add_argument('--loss_name', type=str, default='amsoftmax')
    parser.add_argument('--num_classes', type=int, default=997)
    parser.add_argument('--train_csv_path', type=str, default='data/CN_Celeb1+2_data.csv')
    parser.add_argument('--input_layer', type=str, default='conv2d2')
    parser.add_argument('--pos_enc_layer_type', type=str, default='rel_pos')

    args = parser.parse_args()

    if args.mode == 'enroll':
        if not args.enroll_dir:
            parser.error("--enroll_dir 為 enroll 模式必填")
        enroll(args)
    elif args.mode == 'recognize':
        if not args.recog_lst or not args.enroll_embeddings:
            parser.error("--recog_lst 與 --enroll_embeddings 為 recognize 模式必填")
        recognize(args)

if __name__ == '__main__':
    main()
