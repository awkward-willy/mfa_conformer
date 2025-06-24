import os
import subprocess
from tqdm import tqdm

def convert_flac_to_wav(input_dir):
    """
    Convert all .flac files in the specified directory (and subdirectories) to .wav format
    with a progress bar. Skip conversion if the corresponding .wav file already exists.

    Args:
        input_dir (str): The directory containing .flac files.
    """
    # Gather all .flac files
    flac_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                flac_files.append(os.path.join(root, file))

    # Convert each .flac file to .wav with a progress bar
    for flac_path in tqdm(flac_files, desc="Converting WAV to desire format", unit="file"):
        wav_path = os.path.splitext(flac_path)[0] + "_formatted.wav"

        # Skip conversion if .wav file already exists
        if os.path.exists(wav_path):
            continue

        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", flac_path, "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000", wav_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Error converting: {flac_path}")

if __name__ == "__main__":
    input_directory = "/work/u4701865/downloads/lnflux/Influx_test/wav"
    convert_flac_to_wav(input_directory)
