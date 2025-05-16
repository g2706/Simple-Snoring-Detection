import torch
import os
import librosa
from torch.utils.data import Dataset

class KaggleSnoreDataset(Dataset):
    def __init__(self, root_dir, sr=16000):
        self.sr = sr
        self.samples = []
        self.labels = []

        for label_str in ['0', '1']:
            folder = os.path.join(root_dir, label_str)
            for fname in os.listdir(folder):
                if fname.endswith(".wav"):
                    self.samples.append(os.path.join(folder, fname))
                    self.labels.append(int(label_str))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        y, _ = librosa.load(self.samples[idx], sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13).T  # [T, F]
        label_seq = torch.full((mfcc.shape[0],), float(self.labels[idx]))  # 全部帧标为 0 或 1
        return torch.tensor(mfcc, dtype=torch.float32), label_seq
