import librosa
import numpy as np

def extract_features(audio_path, sr=16000, n_mfcc=13, hop_length=512):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length).T  # [T, F]
    return mfcc, sr, len(y), hop_length
