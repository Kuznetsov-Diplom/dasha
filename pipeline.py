# pipeline.py
# Улучшенный пайплайн: 39-мерные признаки (MFCC + Delta + Delta-Delta) + CMVN

from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
from cv_ru_loader import CommonVoiceRULoader
from feature_normalizer import FeatureNormalizer

class AudioPipeline:
    """Улучшенный пайплайн (39 признаков + CMVN)"""
    SAMPLE_RATE = 16000
    PRE_EMPHASIS_ALPHA = 0.97
    FRAME_LENGTH_MS = 25
    FRAME_SHIFT_MS = 10
    N_MFCC = 13          # базовые MFCC
    N_FEATURES = 39      # 13 MFCC + 13 Delta + 13 Delta-Delta

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

    def _extract_mfcc_full(self, y: np.ndarray) -> np.ndarray:
        """Извлечение 39-мерных признаков (MFCC + Delta + Delta-Delta)"""
        frame_length = int(self.FRAME_LENGTH_MS * self.SAMPLE_RATE / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * self.SAMPLE_RATE / 1000)

        mfcc = librosa.feature.mfcc(
            y=y, sr=self.SAMPLE_RATE, n_mfcc=self.N_MFCC,
            n_fft=frame_length, hop_length=hop_length, window="hamming"
        )
        # Добавляем дельты
        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)

        full_features = np.vstack([mfcc, delta, delta2])  # (39, n_frames)
        return full_features

    def _cmvn(self, features: np.ndarray) -> np.ndarray:
        """Cepstral Mean and Variance Normalization"""
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True) + 1e-8
        return (features - mean) / std

    def extract_raw_39(self, audio_path: str | Path) -> np.ndarray:
        """Извлечение 39-мерного вектора (mean по 39 признакам)"""
        y, _ = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)
        y = self._pre_emphasis(y)
        y = y / (np.max(np.abs(y)) + 1e-8)

        frame_length = int(self.FRAME_LENGTH_MS * self.SAMPLE_RATE / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * self.SAMPLE_RATE / 1000)
        vad_mask = self._vad_energy(y, frame_length, hop_length)

        features = self._extract_mfcc_full(y)          # (39, n_frames)
        active = features[:, vad_mask] if np.any(vad_mask) else features
        active = self._cmvn(active)                    # CMVN

        mean_vec = np.mean(active, axis=1)             # 39-мерный вектор
        return mean_vec

    def extract_features_detailed(self, audio_path: str | Path) -> Tuple[np.ndarray, Dict]:
        raw_39 = self.extract_raw_39(audio_path)
        normalized_39 = self.normalizer.transform(raw_39)

        y, _ = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)
        y_pre = self._pre_emphasis(y)
        y_norm = y_pre / (np.max(np.abs(y_pre)) + 1e-8)

        frame_length = int(self.FRAME_LENGTH_MS * self.SAMPLE_RATE / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * self.SAMPLE_RATE / 1000)
        vad_mask = self._vad_energy(y_norm, frame_length, hop_length)
        mfcc_full = self._extract_mfcc_full(y_norm)

        steps = {
            "original_waveform": y,
            "pre_emphasis_waveform": y_pre,
            "normalized_amplitude_waveform": y_norm,
            "vad_mask": vad_mask,
            "mfcc": mfcc_full[:13],   # только базовые MFCC для визуализации
            "raw_39_vector": raw_39,
            "normalized_39_vector": normalized_39
        }
        return normalized_39, {"steps": steps}

def process_phrase(audio_path: str) -> Dict[str, Any]:
    if not audio_path or not Path(audio_path).exists():
        raise ValueError(f"Файл не найден: {audio_path}")

    pipeline = AudioPipeline()
    normalized_39, details = pipeline.extract_features_detailed(audio_path)
    steps = details["steps"]

    def create_plot(data, title):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(data)
        ax.set_title(title)
        ax.set_xlabel("Сэмплы")
        ax.set_ylabel("Амплитуда")
        plt.close(fig)
        return fig

    return {
        "waveform_plot": create_plot(steps["original_waveform"], "Оригинальный waveform"),
        "pre_emphasis_plot": create_plot(steps["pre_emphasis_waveform"], "После Pre-emphasis"),
        "mfcc_plot": create_plot(steps["mfcc"][0], "MFCC (первый коэффициент)"),
        "normalized_vector": normalized_39.tolist()   # теперь 39-мерный
    }