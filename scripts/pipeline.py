from pathlib import Path
import numpy as np
import librosa
from typing import Tuple, Dict, Any
import json

from .cv_ru_loader import CommonVoiceRULoader
from .feature_normalizer import FeatureNormalizer


class AudioPipeline:
    """Полный аудио-пайплайн строго по Главе 2 дипломной работы.
    
    Реализует:
    - 16 кГц
    - Pre-emphasis (α=0.97)
    - Нормализация амплитуды
    - VAD (energy-based)
    - Фреймирование: Hamming 25мс / 10мс overlap
    - Извлечение 13 MFCC
    - Агрегация mean + std → 26-мерный вектор
    """

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
        """Высокочастотный фильтр (pre-emphasis)."""
        return np.append(y[0], y[1:] - AudioPipeline.PRE_EMPHASIS_ALPHA * y[:-1])

    @staticmethod
    def _vad_energy(y: np.ndarray, frame_length: int, hop_length: int, threshold: float = 0.015) -> np.ndarray:
        """Детекция речевой активности по энергии."""
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        return energy > threshold

    def extract_raw_26(self, audio_path: str | Path) -> np.ndarray:
        """Извлекает СЫРОЙ 26-мерный вектор (13 mean + 13 std) — для обучения нормализатора."""
        y, sr = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)

        # Pre-emphasis
        y = self._pre_emphasis(y)

        # Нормализация амплитуды
        y = y / (np.max(np.abs(y)) + 1e-8)

        frame_length = int(self.FRAME_LENGTH_MS * self.SAMPLE_RATE / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * self.SAMPLE_RATE / 1000)

        # VAD
        vad_mask = self._vad_energy(y, frame_length, hop_length)

        if not np.any(vad_mask):
            # fallback — весь сигнал
            vad_mask = np.ones(len(y) > 0, dtype=bool)

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.SAMPLE_RATE,
            n_mfcc=self.N_MFCC,
            n_fft=frame_length,
            hop_length=hop_length,
            window="hamming"
        )

        # Только активные фреймы
        active_mfcc = mfcc[:, vad_mask] if vad_mask.any() else mfcc

        # 13 mean + 13 std = 26
        mean_vec = np.mean(active_mfcc, axis=1)
        std_vec = np.std(active_mfcc, axis=1, ddof=1)
        raw_features = np.concatenate([mean_vec, std_vec])  # shape: (26,)

        return raw_features

    def extract_features_detailed(self, audio_path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Полное извлечение + все промежуточные данные для визуализаций."""
        raw_26 = self.extract_raw_26(audio_path)
        normalized = self.normalizer.transform(raw_26)

        # Для демо/визуализаций загружаем ещё раз
        y, sr = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)
        y_pre = self._pre_emphasis(y)
        y_norm = y_pre / (np.max(np.abs(y_pre)) + 1e-8)

        frame_length = int(self.FRAME_LENGTH_MS * self.SAMPLE_RATE / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * self.SAMPLE_RATE / 1000)

        # VAD + Energy для визуализации
        energy = librosa.feature.rms(y=y_norm, frame_length=frame_length, hop_length=hop_length)[0]
        vad_mask = self._vad_energy(y_norm, frame_length, hop_length)

        mfcc = librosa.feature.mfcc(
            y=y_norm, sr=self.SAMPLE_RATE, n_mfcc=self.N_MFCC,
            n_fft=frame_length, hop_length=hop_length, window="hamming"
        )

        meta = {
            "n_active_frames": int(np.sum(vad_mask)),
            "dim": len(normalized)
        }

        steps = {
            "original_waveform": y,
            "pre_emphasis_waveform": y_pre,
            "normalized_amplitude": y_norm,
            "vad_mask": vad_mask,
            "energy": energy,                    # ← КРИТИЧНО ДЛЯ Gradio
            "mfcc": mfcc,
            "raw_26_vector": raw_26,
            "normalized_26_vector": normalized
        }

        return normalized, {"steps": steps, "meta": meta}


if __name__ == "__main__":
    pipe = AudioPipeline()
    print("AudioPipeline инициализирован успешно (26-dim по Главе 2)")
