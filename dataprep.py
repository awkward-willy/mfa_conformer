import os
import subprocess
from tqdm import tqdm

def convert_mp4_to_wav(input_dir):
    """
    Convert all .mp4 files in the specified directory (and subdirectories) to .wav format
    with a progress bar.

    Args:
        input_dir (str): The directory containing .mp4 files.
    """
    # Gather all .mp4 files
    mp4_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))

    # Convert each .mp4 file to .wav with a progress bar
    for mp4_path in tqdm(mp4_files, desc="Converting MP4 to WAV", unit="file"):
        wav_path = os.path.splitext(mp4_path)[0] + ".wav"
        # Skip conversion if .wav file already exists
        if os.path.exists(wav_path):
            continue
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", mp4_path, "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000", wav_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Error converting: {mp4_path}")

if __name__ == "__main__":
    input_directory = "/work/u4701865/downloads/Voxceleb/video/vox2_test_mp4"
    convert_mp4_to_wav(input_directory)

