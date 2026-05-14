# scripts/feature_normalizer.py

from pathlib import Path
import numpy as np
import json
from typing import List

class FeatureNormalizer:
    """Z-score нормализатор (mean / std) — рекомендуется в п. 2.6.1 диплома."""

    def __init__(self, params_path: str = "models/audio_params/normalizer_params.json"):
        self.params_path = Path(params_path)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self._load_params()

    def _load_params(self):
        """Загружает параметры. Если файл старый (MinMax) или повреждён — автоматически удаляет его."""
        if self.params_path.exists():
            try:
                with open(self.params_path, encoding="utf-8") as f:
                    data = json.load(f)
                if "mean" in data and "std" in data:
                    self.mean = np.array(data["mean"])
                    self.std = np.array(data["std"])
                    print(f"✅ FeatureNormalizer загружен (Z-score из {self.params_path})")
                else:
                    # Старый MinMax-файл
                    print(f"🗑️  Старый normalizer_params.json (MinMax) удалён")
                    self.params_path.unlink()
                    self.mean = None
                    self.std = None
            except Exception as e:
                print(f"🗑️  Повреждённый normalizer_params.json удалён: {e}")
                if self.params_path.exists():
                    self.params_path.unlink()
                self.mean = None
                self.std = None
        else:
            print("normalizer_params.json не найден — нормализация отключена (identity)")

    def fit(self, features_list: List[np.ndarray]):
        """Обучаем Z-нормализатор на сырых 26-мерных векторах."""
        all_features = np.vstack(features_list)  # (N, 26)
        self.mean = all_features.mean(axis=0)
        self.std = all_features.std(axis=0, ddof=1)
        # Защита от нулевого std
        self.std[self.std < 1e-8] = 1.0

        self.params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.params_path, "w", encoding="utf-8") as f:
            json.dump({
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "n_samples": len(features_list)
            }, f, ensure_ascii=False, indent=2)
        print(f"Z-нормализатор успешно обучен на {len(features_list)} примерах и сохранён")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Применяем Z-score нормализацию."""
        if self.mean is None or self.std is None:
            return features
        return (features - self.mean) / self.std

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Для отладки и визуализации."""
        if self.mean is None or self.std is None:
            return normalized
        return normalized * self.std + self.mean
