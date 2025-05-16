# inference.py
import os

import librosa
import torch
import numpy as np
from cryptography.utils import CryptographyDeprecationWarning
from matplotlib import pyplot as plt

from model import SnoreNet

import warnings
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)


def run_chunked_detection(audio_path, model_path, chunk_sec, threshold):
    y, sr = librosa.load(audio_path, sr=16000)
    model = SnoreNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    chunk_len = int(chunk_sec * sr)
    timestamps = []
    for i in range(0, len(y) - chunk_len + 1, chunk_len):
        chunk = y[i:i+chunk_len]
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13).T  # [T, F]
        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, T, F]
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).squeeze().mean().item()
        if prob > threshold:
            timestamps.append((i / sr, (i + chunk_len) / sr))

    # 合并连续或重叠片段
    merged = []
    for ts in timestamps:
        if not merged:
            merged.append(ts)
        else:
            prev_start, prev_end = merged[-1]
            curr_start, curr_end = ts
            if curr_start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, curr_end))
            else:
                merged.append(ts)

    os.makedirs("img", exist_ok=True)

    print("Detected snore segments (seconds):")
    # for start, end in merged:

    for idx, (start, end) in enumerate(merged, 1):
        duration = end - start
        print(f"{idx:2d}. Start: {start:.2f} sec, Duration: {duration:.2f} sec")

    # 可视化波形 + 鼾声段落
    plt.figure(figsize=(14, 4))
    times = np.arange(len(y)) / sr
    plt.plot(times, y, label="Waveform", alpha=0.7)

    for start, end in merged:
        plt.axvspan(start, end, color='red', alpha=0.3, label='Snore' if start == merged[0][0] else None)

    plt.title("Snore Detection Result")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"img/snore_detection_style.png")
    plt.show()