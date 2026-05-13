# scripts/pipeline.py
from pathlib import Path
import numpy as np
import librosa
from typing import Tuple, Dict, Any
import json

from .cv_ru_loader import CommonVoiceRULoader

class AudioPipeline:
    """Полная обработка речевого сигнала строго по Главе 2 дипломной работы.
    Используем librosa для стабильной загрузки MP3 без torchcodec."""

    SAMPLE_RATE = 16000
    PRE_EMPHASIS_ALPHA = 0.97
    FRAME_LENGTH_MS = 25
    FRAME_SHIFT_MS = 10
    N_MFCC = 13

    def __init__(self):
        self.loader = CommonVoiceRULoader()
        self.params_dir = Path("models/audio_params")
        self.params_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pre_emphasis(y: np.ndarray) -> np.ndarray:
        """Высокочастотный фильтр (pre-emphasis)."""
        return np.append(y[0], y[1:] - AudioPipeline.PRE_EMPHASIS_ALPHA * y[:-1])

    @staticmethod
    def _vad_energy(y: np.ndarray, frame_length: int, hop_length: int, threshold: float = 0.02) -> np.ndarray:
        """Простой energy-based VAD."""
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        return energy > threshold

    @staticmethod
    def extract_features(audio_path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Полный пайплайн Главы 2 → 26-мерный вектор (13 mean + 13 std)."""
        # 1. Загрузка + ресэмпл на 16 кГц (librosa — стабильнее torchaudio для MP3)
        y, sr = librosa.load(str(audio_path), sr=AudioPipeline.SAMPLE_RATE, mono=True)

        # 2. Pre-emphasis
        y = AudioPipeline._pre_emphasis(y)

        # 3. Нормализация амплитуды
        y = y / (np.max(np.abs(y)) + 1e-8)

        # 4. Параметры фрейминга
        frame_length = int(AudioPipeline.FRAME_LENGTH_MS * AudioPipeline.SAMPLE_RATE / 1000)
        hop_length = int(AudioPipeline.FRAME_SHIFT_MS * AudioPipeline.SAMPLE_RATE / 1000)

        # 5. VAD
        vad_mask = AudioPipeline._vad_energy(y, frame_length, hop_length)

        # 6. Извлечение 13 MFCC (окно Хэмминга)
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=AudioPipeline.SAMPLE_RATE,
            n_mfcc=AudioPipeline.N_MFCC,
            n_fft=frame_length,
            hop_length=hop_length,
            window="hamming"
        ).T  # (n_frames, 13)

        # 7. Только активные фреймы
        active_mfcc = mfcc[vad_mask]

        if len(active_mfcc) == 0:
            active_mfcc = mfcc  # fallback

        # 8. Агрегация → 26-мерный вектор
        mean_vec = np.mean(active_mfcc, axis=0)
        std_vec = np.std(active_mfcc, axis=0)
        features = np.concatenate([mean_vec, std_vec])  # 26-dim

        # 9. Сохраняем параметры нормализации (для Главы 3)
        params = {
            "feature_mean": mean_vec.tolist(),
            "feature_std": std_vec.tolist(),
            "n_active_frames": int(len(active_mfcc)),
            "sample_rate": AudioPipeline.SAMPLE_RATE,
        }
        params_path = Path("models/audio_params/audio_normalization_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

        meta = {
            "n_active_frames": int(len(active_mfcc)),
            "params_path": str(params_path),
        }

        return features, meta