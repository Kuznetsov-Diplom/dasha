# scripts/pipeline.py
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import json

from .cv_ru_loader import CommonVoiceRULoader
from .feature_normalizer import FeatureNormalizer   # ← из предыдущего сообщения

class AudioPipeline:
    """Полный пайплайн Главы 2 + пошаговая визуализация для демо."""

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

    def _pre_emphasis(self, y: np.ndarray) -> np.ndarray:
        return np.append(y[0], y[1:] - self.PRE_EMPHASIS_ALPHA * y[:-1])

    def _vad_energy(self, y: np.ndarray, frame_length: int, hop_length: int, threshold: float = 0.02) -> np.ndarray:
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        return energy > threshold

    def extract_features_detailed(self, audio_path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Возвращает финальный вектор + словарь с промежуточными данными для визуализаций."""
        y, sr = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)

        steps = {
            "original_waveform": y.copy(),
            "sample_rate": sr,
        }

        # 1. Pre-emphasis
        y_pre = self._pre_emphasis(y)
        steps["pre_emphasis_waveform"] = y_pre.copy()

        # 2. Нормализация амплитуды
        y_norm = y_pre / (np.max(np.abs(y_pre)) + 1e-8)
        steps["normalized_waveform"] = y_norm.copy()

        # 3. Фрейминг + VAD
        frame_length = int(self.FRAME_LENGTH_MS * self.SAMPLE_RATE / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * self.SAMPLE_RATE / 1000)
        vad_mask = self._vad_energy(y_norm, frame_length, hop_length)
        steps["vad_mask"] = vad_mask.copy()
        steps["energy"] = librosa.feature.rms(y=y_norm, frame_length=frame_length, hop_length=hop_length)[0]

        # 4. MFCC
        mfcc = librosa.feature.mfcc(
            y=y_norm, sr=self.SAMPLE_RATE, n_mfcc=self.N_MFCC,
            n_fft=frame_length, hop_length=hop_length, window="hamming"
        )
        steps["mfcc"] = mfcc.copy()                     # shape (13, n_frames)

        active_mfcc = mfcc[:, vad_mask] if np.any(vad_mask) else mfcc
        mean_vec = np.mean(active_mfcc, axis=1)
        std_vec = np.std(active_mfcc, axis=1)
        raw_features = np.concatenate([mean_vec, std_vec])

        # 5. Финальная нормализация [0,1] (для Главы 3)
        normalized_features = self.normalizer.transform(raw_features)

        # Сохраняем параметры
        params = {"normalized_features": normalized_features.tolist(), "n_active_frames": int(len(active_mfcc[0]))}
        params_path = self.params_dir / "audio_normalization_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

        steps["raw_26_vector"] = raw_features
        steps["normalized_26_vector"] = normalized_features

        meta = {
            "n_active_frames": int(len(active_mfcc[0])),
            "params_path": str(params_path),
            "normalized_to": "[0, 1]"
        }

        return normalized_features, {"steps": steps, "meta": meta}