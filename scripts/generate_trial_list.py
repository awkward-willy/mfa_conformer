import os
import argparse
from itertools import combinations, product
from tqdm import tqdm  # 進度條顯示

def build_trial_list(dataset_dir, output_path):
    """
    產生 speaker verification 的 trial list。
    
    同一資料夾內的檔案視為同一語者，不同資料夾的檔案視為不同語者。
    標籤規則：
        1 代表同一語者
        0 代表不同語者

    Args:
        dataset_dir (str): 資料集根目錄
        output_path (str): 輸出檔案路徑
    """
    # 收集每個語者(資料夾)內所有 wav 檔案
    speaker_dict = {}
    print("收集每個語者的音檔...")
    for root, _, files in tqdm(os.walk(dataset_dir), desc="掃描資料夾", unit="dir"):
        for file in files:
            if file.endswith(".wav"):
                # 語者ID即為當前資料夾名稱
                speaker_id = os.path.basename(root)
                if speaker_id not in speaker_dict:
                    speaker_dict[speaker_id] = []
                speaker_dict[speaker_id].append(os.path.join(root, file))
    
    same_speaker_pairs = []
    diff_speaker_pairs = []

    # 產生同一語者配對 (標籤 0)
    print("產生同一語者配對...")
    for speaker_id, files in tqdm(speaker_dict.items(), desc="同一語者配對", unit="speaker"):
        if len(files) > 1:
            # 若有重複的檔名，可依需求決定是否過濾，這裡直接產生所有可能配對
            pairs = list(combinations(files, 2))
            same_speaker_pairs.extend(pairs)

    # 產生不同語者配對 (標籤 1)
    print("產生不同語者配對...")
    speakers = list(speaker_dict.keys())
    for spk1, spk2 in tqdm(list(combinations(speakers, 2)), desc="不同語者配對", unit="pair"):
        pairs = list(product(speaker_dict[spk1], speaker_dict[spk2]))
        diff_speaker_pairs.extend(pairs)

    # 寫入 trial list 檔案
    print("儲存 trial list 到檔案...")
    with open(output_path, "w") as f:
        for file1, file2 in tqdm(same_speaker_pairs, desc="寫入同一語者配對", unit="pair"):
            f.write(f"1 {file1} {file2}\n")
        for file1, file2 in tqdm(diff_speaker_pairs, desc="寫入不同語者配對", unit="pair"):
            f.write(f"0 {file1} {file2}\n")
    
    print(f"Trial list 已儲存到 {output_path}")
    print(f"總共產生 {len(same_speaker_pairs)} 個同一語者配對和 {len(diff_speaker_pairs)} 個不同語者配對.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="產生 speaker verification 的 trial list (0: 同一語者, 1: 不同語者)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="資料集根目錄路徑")
    parser.add_argument("--output_path", type=str, required=True, help="輸出 trial list 檔案的路徑")
    args = parser.parse_args()

    build_trial_list(args.dataset_dir, args.output_path)
