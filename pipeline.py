# pipeline.py
# Чистая финальная версия

from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

class AudioPipeline:
    SAMPLE_RATE = 16000
    PRE_EMPHASIS_ALPHA = 0.97
    FRAME_LENGTH_MS = 25
    FRAME_SHIFT_MS = 10
    N_MFCC = 13

    def __init__(self):
        pass

    @staticmethod
    def _pre_emphasis(y: np.ndarray) -> np.ndarray:
        return np.append(y[0], y[1:] - AudioPipeline.PRE_EMPHASIS_ALPHA * y[:-1])

    @staticmethod
    def _vad_energy(y: np.ndarray, frame_length: int, hop_length: int, threshold: float = 0.02) -> np.ndarray:
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        return energy > threshold

    def extract_features(self, audio_path: str) -> Dict:
        y, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)

        # Pre-emphasis
        y_pre = self._pre_emphasis(y)

        # VAD
        frame_length = int(self.FRAME_LENGTH_MS * sr / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * sr / 1000)
        vad_mask = self._vad_energy(y_pre, frame_length, hop_length)

        # MFCC + Delta + Delta-Delta (39 признаков)
        mfcc = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=self.N_MFCC,
                                    n_fft=frame_length, hop_length=hop_length, window="hamming")
        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features_39 = np.vstack([mfcc, delta, delta2])

        # Активные кадры
        active = features_39[:, vad_mask] if np.any(vad_mask) else features_39

        # Средний вектор (39-dim)
        mean_vec = np.mean(active, axis=1)

        return {
            "y_orig": y,
            "y_pre": y_pre,
            "sr": sr,
            "vad_mask": vad_mask,
            "mfcc": mfcc,
            "features_39": features_39,
            "normalized_vector": mean_vec
        }

def process_phrase(audio_path: str) -> Dict[str, Any]:
    if not audio_path or not Path(audio_path).exists():
        raise ValueError(f"Файл не найден: {audio_path}")

    pipeline = AudioPipeline()
    data = pipeline.extract_features(audio_path)

    return {
        "y_orig": data["y_orig"],
        "sr": data["sr"],
        "mfcc": data["mfcc"],
        "normalized_vector": data["normalized_vector"].tolist()
    }