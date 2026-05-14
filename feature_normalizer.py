# feature_normalizer.py
from pathlib import Path
import numpy as np
import json
from typing import List

class FeatureNormalizer:
    """Z-нормализатор (стандартное масштабирование: mean=0, std=1) для 26-мерного вектора
    признаков строго по разделу 2.6.1 дипломной работы.
    Сохраняет параметры mean/std по каждому измерению для последующего использования
    на этапах регистрации и верификации (воспроизводимость).
    """

    def __init__(self, params_path: str = "models/audio_params/normalizer_params.json"):
        self.params_path = Path(params_path)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self._load_params()

    def _load_params(self):
        if self.params_path.exists():
            with open(self.params_path, encoding="utf-8") as f:
                data = json.load(f)
            self.mean = np.array(data["mean"])
            self.std = np.array(data["std"])
            print(f"✅ FeatureNormalizer (Z-score) загружен из {self.params_path}")
        else:
            print("⚠️ normalizer_params.json не найден — нормализация отключена")

    def fit(self, features_list: List[np.ndarray]):
        """Обучаем на списке 26-мерных сырых векторов (вызывать один раз на обучающей выборке)."""
        if not features_list:
            raise ValueError("features_list пуст")
        all_features = np.vstack(features_list)  # (N, 26)
        self.mean = all_features.mean(axis=0)
        self.std = all_features.std(axis=0, ddof=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)  # защита от деления на 0

        self.params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.params_path, "w", encoding="utf-8") as f:
            json.dump({
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "n_samples": len(features_list),
                "description": "Z-normalization parameters (mean/std) по Главе 2"
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ Z-нормализатор обучен на {len(features_list)} примерах и сохранён в {self.params_path}")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Применяем Z-нормализацию."""
        if self.mean is None or self.std is None:
            return features
        return (features - self.mean) / self.std

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Обратная трансформация (для отладки и визуализаций)."""
        if self.mean is None or self.std is None:
            return normalized
        return normalized * self.std + self.mean
