# dasha/audio/pipeline.py
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torchaudio
import soundfile as sf

class DashaAudioPipeline:
    """
    Полный пайплайн Главы 2 (строго по тексту диплома):
      • 16 кГц
      • Pre-emphasis (0.97)
      • Нормализация амплитуды
      • Окно Хэмминга 25 мс, сдвиг 10 мс
      • Frame-aligned VAD
      • 13 MFCC
      • mean + std по активным фреймам → 26-мерный вектор
      • Сохранение параметров нормализации
    """

    SAMPLE_RATE = 16000
    FRAME_LENGTH = 0.025
    HOP_LENGTH = 0.010
    PRE_EMPHASIS_COEF = 0.97
    N_MFCC = 13

    def __init__(self, models_dir: str = "models/audio_params"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self._load_normalizer()

    def _load_normalizer(self):
        mean_path = self.models_dir / "feature_mean.npy"
        std_path = self.models_dir / "feature_std.npy"
        if mean_path.exists() and std_path.exists():
            self.feature_mean = np.load(mean_path)
            self.feature_std = np.load(std_path)
            print(f"✅ Загружены параметры нормализации (shape={self.feature_mean.shape})")
        else:
            print("ℹ️ Параметры нормализации не найдены — будут созданы при первом запуске")

    def _save_normalizer(self, mean: np.ndarray, std: np.ndarray):
        """Всегда сохраняем 26-мерный вектор"""
        np.save(self.models_dir / "feature_mean.npy", mean)
        np.save(self.models_dir / "feature_std.npy", std)
        self.feature_mean = mean
        self.feature_std = std
        print(f"✅ Параметры нормализации сохранены (shape={mean.shape})")

    def extract_features(self, audio_path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        audio_path = Path(audio_path)

        # Загрузка через soundfile (стабильно, без torchcodec)
        sig, sr = sf.read(str(audio_path), dtype='float32')
        if len(sig.shape) > 1:
            sig = np.mean(sig, axis=1)

        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            sig = resampler(torch.from_numpy(sig)).numpy()

        # 1. Pre-emphasis
        sig = np.append(sig[0], sig[1:] - self.PRE_EMPHASIS_COEF * sig[:-1])

        # 2. Нормализация амплитуды
        sig = sig / (np.max(np.abs(sig)) + 1e-8)

        # 3. Параметры фреймирования
        frame_len = int(self.FRAME_LENGTH * self.SAMPLE_RATE)
        hop_len = int(self.HOP_LENGTH * self.SAMPLE_RATE)

        # MFCC с окном Хэмминга
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.SAMPLE_RATE,
            n_mfcc=self.N_MFCC,
            melkwargs={
                "n_fft": 512,
                "win_length": frame_len,
                "hop_length": hop_len,
                "window_fn": torch.hamming_window,
                "n_mels": 40,
            }
        )

        sig_torch = torch.from_numpy(sig).unsqueeze(0)
        mfcc = mfcc_transform(sig_torch).squeeze(0).numpy()   # (13, T)

        # Frame-aligned VAD (энергия + zero-crossing)
        energy = np.sum(mfcc**2, axis=0)
        zcr = np.sum(np.abs(np.diff(np.sign(sig.reshape(-1, 1)), axis=0)), axis=0)[:mfcc.shape[1]]
        vad_mask = (energy > np.percentile(energy, 20)) & (zcr > np.percentile(zcr, 25))

        active_frames = mfcc[:, vad_mask]
        if active_frames.shape[1] == 0:  # fallback
            active_frames = mfcc
            print("⚠️ VAD не нашёл активных фреймов — используем все")

        # 4. 26-мерный вектор
        feat_mean = np.mean(active_frames, axis=1)
        feat_std = np.std(active_frames, axis=1)
        features = np.concatenate([feat_mean, feat_std])   # 26-dim

        # Сохранение / применение нормализации
        if self.feature_mean is None or self.feature_mean.shape[0] != 26:
            self._save_normalizer(feat_mean, feat_std)
        else:
            features = (features - self.feature_mean) / (self.feature_std + 1e-8)

        meta = {
            "duration_sec": len(sig) / self.SAMPLE_RATE,
            "active_ratio": active_frames.shape[1] / mfcc.shape[1],
            "n_active_frames": int(active_frames.shape[1]),
            "total_frames": int(mfcc.shape[1]),
        }

        print(f"✅ 26-мерный вектор | Активно: {meta['active_ratio']*100:.1f}% "
              f"({meta['n_active_frames']}/{meta['total_frames']} фреймов)")

        return features.astype(np.float32), meta


# Singleton
pipeline = DashaAudioPipeline()