# pipeline.py
# Финальная версия с поддержкой переобучения нормализатора

from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
from cv_ru_loader import CommonVoiceRULoader
from feature_normalizer import FeatureNormalizer

class AudioPipeline:
    SAMPLE_RATE = 16000
    PRE_EMPHASIS_ALPHA = 0.97
    FRAME_LENGTH_MS = 25
    FRAME_SHIFT_MS = 10
    N_MFCC = 13
    N_FEATURES = 39

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
        frame_length = int(self.FRAME_LENGTH_MS * self.SAMPLE_RATE / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * self.SAMPLE_RATE / 1000)
        mfcc = librosa.feature.mfcc(y=y, sr=self.SAMPLE_RATE, n_mfcc=self.N_MFCC,
                                    n_fft=frame_length, hop_length=hop_length, window="hamming")
        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        return np.vstack([mfcc, delta, delta2])

    def extract_raw_39(self, audio_path: str | Path) -> np.ndarray:
        y, _ = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)
        y = self._pre_emphasis(y)
        y = y / (np.max(np.abs(y)) + 1e-8)

        frame_length = int(self.FRAME_LENGTH_MS * self.SAMPLE_RATE / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * self.SAMPLE_RATE / 1000)
        vad_mask = self._vad_energy(y, frame_length, hop_length)

        features = self._extract_mfcc_full(y)
        active = features[:, vad_mask] if np.any(vad_mask) else features

        mean_vec = np.mean(active, axis=1)
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
            "mfcc": mfcc_full[:13],
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
        "normalized_vector": normalized_39.tolist()
    }

# ====================== ФУНКЦИЯ ПЕРЕОБУЧЕНИЯ НОРМАЛИЗАТОРА ======================
def retrain_normalizer(max_speakers: int = 100, phrases_per_speaker: int = 8):
    """Переобучает нормализатор на 39-мерных признаках"""
    from cv_ru_loader import load_speakers_and_phrases
    import json

    speakers = load_speakers_and_phrases()
    pipeline = AudioPipeline()
    all_vectors = []

    speaker_list = list(speakers.keys())[:max_speakers]

    for sp in speaker_list:
        for item in speakers[sp]['phrases'][:phrases_per_speaker]:
            try:
                vec = pipeline.extract_raw_39(item['audio_path'])
                all_vectors.append(vec)
            except:
                continue

    if len(all_vectors) < 50:
        return False, f"Слишком мало данных ({len(all_vectors)} векторов)"

    all_vectors = np.array(all_vectors)
    pipeline.normalizer.fit(all_vectors)

    # Сохраняем новые параметры
    params = {
        "min_val": pipeline.normalizer.min_val.tolist(),
        "max_val": pipeline.normalizer.max_val.tolist(),
        "feature_dim": 39
    }
    with open(pipeline.params_dir / "normalizer_params.json", "w") as f:
        json.dump(params, f)

    return True, f"Нормализатор успешно переобучен на {len(all_vectors)} векторах (39 признаков)"