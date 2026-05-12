"""
Dasha Audio Pipeline — Глава 2
Строго по требованиям дипломной работы
"""

from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf
from typing import Tuple, Dict

class DashaAudioPipeline:
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.params_dir = Path("models/audio_params")
        self.params_dir.mkdir(parents=True, exist_ok=True)

        self.frame_length = int(0.025 * self.sr)  # 400
        self.hop_length = int(0.010 * self.sr)    # 160
        self.n_mfcc = 13

        self.feature_mean = None
        self.feature_std = None

    def pre_emphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        return np.append(audio[0], audio[1:] - coef * audio[:-1])

    def normalize_amplitude(self, audio: np.ndarray) -> np.ndarray:
        max_amp = np.max(np.abs(audio))
        return audio / (max_amp + 1e-8) if max_amp > 0 else audio

    def simple_vad(self, audio: np.ndarray, threshold_db: float = -35) -> np.ndarray:
        """Energy-based VAD"""
        energy = 20 * np.log10(np.std(np.lib.stride_tricks.sliding_window_view(
            audio, self.frame_length)[::self.hop_length//2], axis=1) + 1e-8)
        return energy > threshold_db

    def extract_features(self, audio_path: str | Path) -> Tuple[np.ndarray, Dict]:
        audio, sr = sf.read(audio_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != self.sr:
            audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, self.sr).numpy()

        audio = self.pre_emphasis(audio)
        audio = self.normalize_amplitude(audio)

        # VAD
        active = self.simple_vad(audio)

        # MFCC
        waveform = torch.from_numpy(audio).unsqueeze(0)
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sr,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_fft": 512, "hop_length": self.hop_length, "win_length": self.frame_length, "n_mels": 80}
        )
        mfcc = mfcc_transform(waveform).squeeze(0).numpy()  # [13, T]

        # Активные фреймы
        active_mask = np.repeat(active, mfcc.shape[1] // len(active) + 1)[:mfcc.shape[1]]
        active_mfcc = mfcc[:, active_mask] if active_mask.any() else mfcc

        # 26-мерный вектор
        feat_mean = np.mean(active_mfcc, axis=1)
        feat_std = np.std(active_mfcc, axis=1)
        features = np.concatenate([feat_mean, feat_std])

        # Сохраняем параметры
        if self.feature_mean is None:
            self.feature_mean = feat_mean
            self.feature_std = feat_std
            np.save(self.params_dir / "feature_mean.npy", feat_mean)
            np.save(self.params_dir / "feature_std.npy", feat_std)

        meta = {
            "duration_sec": round(len(audio) / self.sr, 2),
            "active_ratio": round(active_mask.mean(), 3),
        }
        return features, meta


pipeline = DashaAudioPipeline()