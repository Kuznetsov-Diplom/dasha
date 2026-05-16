# feature_normalizer.py
from pathlib import Path
import numpy as np
import json
from typing import List

class FeatureNormalizer:
    """MinMaxScaler в диапазон [0, 1] — именно то, что принимает НПБК по ГОСТ."""

    def __init__(self, params_path: str = "models/audio_params/normalizer_params.json"):
        self.params_path = Path(params_path)
        self.min_val: np.ndarray | None = None
        self.max_val: np.ndarray | None = None
        self._load_params()

    def _load_params(self):
        if self.params_path.exists():
            with open(self.params_path, encoding="utf-8") as f:
                data = json.load(f)
            self.min_val = np.array(data["min"])
            self.max_val = np.array(data["max"])
            print(f"✅ FeatureNormalizer (MinMax [0,1]) загружен из {self.params_path}")
        else:
            print("⚠️ normalizer_params.json не найден — нормализация отключена (будет identity)")

    def fit(self, features_list: List[np.ndarray]):
        """Обучаем MinMaxScaler на списке 26-мерных векторов."""
        if not features_list:
            raise ValueError("features_list пуст")
        all_features = np.vstack(features_list)          # (N, 26)
        self.min_val = all_features.min(axis=0)
        self.max_val = all_features.max(axis=0)

        # Защита от деления на ноль
        range_val = self.max_val - self.min_val
        range_val = np.where(range_val < 1e-8, 1.0, range_val)

        self.params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.params_path, "w", encoding="utf-8") as f:
            json.dump({
                "min": self.min_val.tolist(),
                "max": self.max_val.tolist(),
                "n_samples": len(features_list),
                "description": "MinMax normalization [0,1] для ГОСТ"
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ MinMax-нормализатор обучен на {len(features_list)} примерах и сохранён в {self.params_path}")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Применяем нормализацию в [0, 1]."""
        if self.min_val is None or self.max_val is None:
            return features  # identity, если параметров нет
        range_val = self.max_val - self.min_val
        range_val = np.where(range_val < 1e-8, 1.0, range_val)
        normalized = (features - self.min_val) / range_val
        return np.clip(normalized, 0.0, 1.0)   # жёстко в [0,1]

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Обратная трансформация (для отладки)."""
        if self.min_val is None or self.max_val is None:
            return normalized
        range_val = self.max_val - self.min_val
        range_val = np.where(range_val < 1e-8, 1.0, range_val)
        return normalized * range_val + self.min_val