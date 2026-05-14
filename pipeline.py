# scripts/pipeline.py
from pathlib import Path
import numpy as np
import librosa
from typing import Tuple, Dict, Any
import json

from .cv_ru_loader import CommonVoiceRULoader
from .feature_normalizer import FeatureNormalizer

class AudioPipeline:
    """Пайплайн с delta-MFCC для лучшей разделимости."""

    SAMPLE_RATE = 16000
    PRE_EMPHASIS_ALPHA = 0.97
    FRAME_LENGTH_MS = 25
    FRAME_SHIFT_MS = 10
    N_MFCC = 13

    def __init__(self):
        self.loader = CommonVoiceRULoader()
        self.normalizer = FeatureNormalizer()
        self.params_dir = Path("models/audio_params")
        self.params_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pre_emphasis(y: np.ndarray) -> np.ndarray:
        return np.append(y[0], y[1:] - AudioPipeline.PRE_EMPHASIS_ALPHA * y[:-1])

    @staticmethod
    def _vad_energy(y: np.ndarray, frame_length: int, hop_length: int, threshold: float = 0.018) -> np.ndarray:
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        return energy > threshold

    def extract_features_detailed(self, audio_path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        y, sr = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)

        y = self._pre_emphasis(y)
        y = y / (np.max(np.abs(y)) + 1e-8)

        frame_length = int(self.FRAME_LENGTH_MS * self.SAMPLE_RATE / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * self.SAMPLE_RATE / 1000)

        vad_mask = self._vad_energy(y, frame_length, hop_length)

        mfcc = librosa.feature.mfcc(
            y=y, sr=self.SAMPLE_RATE, n_mfcc=self.N_MFCC,
            n_fft=frame_length, hop_length=hop_length, window="hamming"
        )

        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # 13 static + 13 delta + 13 delta2 = 39 признаков
        full_features = np.vstack([mfcc, delta, delta2])

        active = full_features[:, vad_mask] if np.any(vad_mask) else full_features

        mean_vec = np.mean(active, axis=1)
        std_vec = np.std(active, axis=1)
        raw_features = np.concatenate([mean_vec, std_vec])   # 78-мерный

        normalized = self.normalizer.transform(raw_features)

        meta = {"n_active_frames": int(len(active[0])), "dim": len(normalized)}

        steps = {
            "original_waveform": y,
            "pre_emphasis_waveform": self._pre_emphasis(y),
            "energy": librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0],
            "vad_mask": vad_mask,
            "mfcc": mfcc,
            "raw_26_vector": raw_features[:26],
            "normalized_26_vector": normalized[:26]
        }

        return normalized, {"steps": steps, "meta": meta}