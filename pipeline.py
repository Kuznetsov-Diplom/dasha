# pipeline.py
# Финальная версия с CMVN + L2 нормализацией (исправлено: MinMax для графика ПЕРЕД L2)

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

        y_pre = self._pre_emphasis(y)

        frame_length = int(self.FRAME_LENGTH_MS * sr / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * sr / 1000)
        vad_mask = self._vad_energy(y_pre, frame_length, hop_length)

        mfcc = librosa.feature.mfcc(
            y=y_pre, sr=sr, n_mfcc=self.N_MFCC,
            n_fft=frame_length, hop_length=hop_length, window="hamming"
        )
        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features_39 = np.vstack([mfcc, delta, delta2])

        active = features_39[:, vad_mask] if np.any(vad_mask) else features_39

        # ====================== CMVN + L2 (рекомендация №1) ======================
        # CMVN: убираем среднее и дисперсию по каждому из 39 коэффициентов
        mean_per_dim = np.mean(active, axis=1, keepdims=True)
        std_per_dim = np.std(active, axis=1, keepdims=True) + 1e-8
        active_cmn = (active - mean_per_dim) / std_per_dim

        # Mean pooling (это основа для обоих векторов)
        mean_vec = np.mean(active_cmn, axis=1)

        # Для графика: MinMax в [0,1] на CMVN-векторе (ДО L2) — теперь будет красивый разброс!
        min_v = np.min(mean_vec)
        max_v = np.max(mean_vec)
        if max_v - min_v < 1e-8:
            plot_vec = np.full_like(mean_vec, 0.5)
        else:
            plot_vec = (mean_vec - min_v) / (max_v - min_v)

        # L2-normalization только для реального вектора НПБК
        norm = np.linalg.norm(mean_vec) + 1e-8
        cmvn_l2_vec = mean_vec / norm

        return {
            "y_orig": y,
            "y_pre": y_pre,
            "sr": sr,
            "vad_mask": vad_mask,
            "mfcc": mfcc,
            "features_39": features_39,
            "normalized_vector": plot_vec.tolist(),   # красивый график с разбросом
            "cmvn_vector": cmvn_l2_vec.tolist()          # настоящий сильный вектор для НПБК
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
        "normalized_vector": data["normalized_vector"],
        "cmvn_vector": data["cmvn_vector"]          # ← используй этот для НПБК!
    }

# ====================== ФУНКЦИИ ДЛЯ ВКЛАДКИ 4 ======================
def ensure_normalizer_trained():
    params_path = Path("models/audio_params/normalizer_params.json")
    if params_path.exists():
        return True, "Нормализатор уже существует"

    from cv_ru_loader import load_speakers_and_phrases
    speakers = load_speakers_and_phrases()
    pipeline = AudioPipeline()
    all_vectors = []

    for sp in list(speakers.keys())[:80]:
        for item in speakers[sp]['phrases'][:8]:
            try:
                vec = pipeline.extract_features(item['audio_path'])['cmvn_vector']
                all_vectors.append(vec)
            except:
                continue

    if len(all_vectors) < 30:
        return False, "Недостаточно данных для обучения"

    params = {
        "min_val": [0.0] * 39,
        "max_val": [1.0] * 39,
        "feature_dim": 39
    }
    import json
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_path, "w") as f:
        json.dump(params, f)

    return True, f"Автоматически создан нормализатор на {len(all_vectors)} векторах"

def retrain_normalizer(max_speakers: int = 100, phrases_per_speaker: int = 8):
    from cv_ru_loader import load_speakers_and_phrases
    import json

    speakers = load_speakers_and_phrases()
    pipeline = AudioPipeline()
    all_vectors = []

    speaker_list = list(speakers.keys())[:max_speakers]

    for sp in speaker_list:
        for item in speakers[sp]['phrases'][:phrases_per_speaker]:
            try:
                vec = pipeline.extract_features(item['audio_path'])['cmvn_vector']
                all_vectors.append(vec)
            except:
                continue

    if len(all_vectors) < 50:
        return False, f"Слишком мало данных ({len(all_vectors)} векторов)"

    all_vectors = np.array(all_vectors)

    params = {
        "min_val": [0.0] * 39,
        "max_val": [1.0] * 39,
        "feature_dim": 39
    }
    params_path = Path("models/audio_params/normalizer_params.json")
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_path, "w") as f:
        json.dump(params, f)

    return True, f"Нормализатор успешно переобучен на {len(all_vectors)} векторах (39 признаков)"