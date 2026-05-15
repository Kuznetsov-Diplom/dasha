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

    def _extract_mfcc(self, y: np.ndarray) -> np.ndarray:
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

    def extract_features_detailed(self, audio_path: str | Path) -> Tuple[np.ndarray, Dict]:
        y, _ = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)
        y_pre = self._pre_emphasis(y)
        y_norm = y_pre / (np.max(np.abs(y_pre)) + 1e-8)

        frame_length = int(self.FRAME_LENGTH_MS * self.SAMPLE_RATE / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * self.SAMPLE_RATE / 1000)
        vad_mask = self._vad_energy(y_norm, frame_length, hop_length)

        mfcc = self._extract_mfcc(y_norm)
        active_frames = mfcc[:, vad_mask] if np.any(vad_mask) else mfcc

        raw_26 = np.concatenate([np.mean(active_frames, axis=1), np.std(active_frames, axis=1)])
        normalized_26 = self.normalizer.transform(raw_26)

        steps = {
            "original_waveform": y,
            "pre_emphasis_waveform": y_pre,
            "normalized_amplitude_waveform": y_norm,
            "vad_mask": vad_mask,
            "mfcc": mfcc,
            "raw_26_vector": raw_26,
            "normalized_26_vector": normalized_26
        }
        return normalized_26, {"steps": steps}

def process_phrase(audio_path: str) -> Dict[str, Any]:
    """Главная функция, которую вызывает ui.py"""
    if not audio_path or not Path(audio_path).exists():
        raise ValueError(f"Файл не найден: {audio_path}")

    pipeline = AudioPipeline()
    normalized_26, details = pipeline.extract_features_detailed(audio_path)
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
        "normalized_vector": normalized_26.tolist()
    }