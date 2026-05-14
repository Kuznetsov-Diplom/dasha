# pipeline.py
from pathlib import Path
import numpy as np
import librosa
from typing import Tuple, Dict, Any
from .cv_ru_loader import CommonVoiceRULoader
from .feature_normalizer import FeatureNormalizer

class AudioPipeline:
    """Основной пайплайн предварительной обработки и извлечения признаков строго по Главе 2
    дипломной работы (разделы 2.4, 2.5, 2.6).

    Соответствует требованиям:
    - Частота дискретизации: 16 кГц
    - Pre-emphasis (высокочастотный фильтр)
    - Voice Activity Detection (VAD) по энергии
    - Нормализация амплитуды
    - Фреймирование: окно Хэмминга 25 мс, сдвиг 10 мс
    - Извлечение ровно 13 MFCC-коэффициентов
    - Агрегация: mean + std по 13 MFCC на активных фреймах → 26-мерный вектор
    - Z-нормализация с сохранением параметров (mean/std)
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
        """Pre-emphasis — высокочастотный фильтр (раздел 2.4)."""
        return np.append(y[0], y[1:] - AudioPipeline.PRE_EMPHASIS_ALPHA * y[:-1])

    @staticmethod
    def _vad_energy(y: np.ndarray, frame_length: int, hop_length: int, threshold: float = 0.018) -> np.ndarray:
        """VAD по энергии сигнала (раздел 2.4). Можно улучшить спектральным потоком позже."""
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        return energy > threshold

    def _extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """Извлечение 13 MFCC с заданными параметрами фрейминга."""
        frame_length = int(self.FRAME_LENGTH_MS * self.SAMPLE_RATE / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * self.SAMPLE_RATE / 1000)
        return librosa.feature.mfcc(
            y=y,
            sr=self.SAMPLE_RATE,
            n_mfcc=self.N_MFCC,
            n_fft=frame_length,
            hop_length=hop_length,
            window="hamming"
        )

    def extract_raw_26(self, audio_path: str | Path) -> np.ndarray:
        """Извлекает сырой 26-мерный вектор (mean + std 13 MFCC) — используется для fit нормализатора."""
        y, sr = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)

        # Pre-emphasis
        y = self._pre_emphasis(y)

        # Нормализация амплитуды
        y = y / (np.max(np.abs(y)) + 1e-8)

        frame_length = int(self.FRAME_LENGTH_MS * self.SAMPLE_RATE / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * self.SAMPLE_RATE / 1000)

        # VAD
        vad_mask = self._vad_energy(y, frame_length, hop_length)

        # MFCC
        mfcc = self._extract_mfcc(y)

        # Только активные фреймы
        active_frames = mfcc[:, vad_mask] if np.any(vad_mask) else mfcc

        # Агрегация по Главе 2.6
        mean_vec = np.mean(active_frames, axis=1)  # 13
        std_vec = np.std(active_frames, axis=1)    # 13
        raw_26 = np.concatenate([mean_vec, std_vec])  # 26-dim

        return raw_26

    def extract_features_detailed(self, audio_path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Полный пайплайн: возвращает Z-нормализованный 26-dim вектор + все шаги для визуализаций."""
        # Сырой вектор
        raw_26 = self.extract_raw_26(audio_path)

        # Нормализация
        normalized_26 = self.normalizer.transform(raw_26)

        # Для визуализаций recompute intermediates
        y, _ = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)
        y_pre = self._pre_emphasis(y)
        y_norm = y_pre / (np.max(np.abs(y_pre)) + 1e-8)
        frame_length = int(self.FRAME_LENGTH_MS * self.SAMPLE_RATE / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * self.SAMPLE_RATE / 1000)
        vad_mask = self._vad_energy(y_norm, frame_length, hop_length)
        mfcc = self._extract_mfcc(y_norm)

        n_active = int(np.sum(vad_mask)) if np.any(vad_mask) else mfcc.shape[1]

        meta = {"n_active_frames": n_active, "dim": 26}

        steps = {
            "original_waveform": y,
            "pre_emphasis_waveform": y_pre,
            "normalized_amplitude_waveform": y_norm,
            "energy": librosa.feature.rms(y=y_norm, frame_length=frame_length, hop_length=hop_length)[0],
            "vad_mask": vad_mask,
            "mfcc": mfcc,  # 13 x frames
            "raw_26_vector": raw_26,
            "normalized_26_vector": normalized_26
        }

        return normalized_26, {"steps": steps, "meta": meta}
