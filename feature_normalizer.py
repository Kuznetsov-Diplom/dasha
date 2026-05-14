# scripts/feature_normalizer.py
from pathlib import Path
import numpy as np
import json
from typing import Tuple

class FeatureNormalizer:
    """MinMax-нормализатор 26-мерного вектора к [0, 1] строго по разделу 2.6.1 диплома."""

    def __init__(self, params_path: str = "models/audio_params/normalizer_params.json"):
        self.params_path = Path(params_path)
        self.min_val: np.ndarray | None = None
        self.max_val: np.ndarray | None = None
        self._load_params()

    def _load_params(self):
        if self.params_path.exists():
            with open(self.params_path, encoding="utf-8") as f:
                data = json.load(f)
            self.min_val = np.array(data["min_val"])
            self.max_val = np.array(data["max_val"])
            print(f"✅ FeatureNormalizer загружен (min/max из {self.params_path})")
        else:
            print("⚠️ normalizer_params.json не найден — нормализация отключена")

    def fit(self, features_list: list[np.ndarray]):
        """Фитим на списке 26-мерных векторов (вызывать один раз на обучающей выборке)."""
        all_features = np.vstack(features_list)  # shape: (N, 26)
        self.min_val = all_features.min(axis=0)
        self.max_val = all_features.max(axis=0)
        
        self.params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.params_path, "w", encoding="utf-8") as f:
            json.dump({
                "min_val": self.min_val.tolist(),
                "max_val": self.max_val.tolist(),
                "n_samples": len(features_list)
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ Нормализатор обучен на {len(features_list)} примерах → сохранён в {self.params_path}")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Применяем MinMax [0, 1]."""
        if self.min_val is None or self.max_val is None:
            return features  # fallback
        # Защита от деления на ноль
        range_val = self.max_val - self.min_val
        range_val[range_val == 0] = 1.0
        return (features - self.min_val) / range_val