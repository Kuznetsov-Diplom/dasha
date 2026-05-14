# scripts/fit_normalizer.py
import sys
from pathlib import Path
# Фикс импортов — работает и при запуске python scripts/..., и python -m scripts...
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scripts.cv_ru_loader import CommonVoiceRULoader
from scripts.pipeline import AudioPipeline

def fit_normalizer(n_samples: int = 300):
    """Обучаем FeatureNormalizer на реальных записях датасета.
    После этого 26-мерный вектор всегда будет в диапазоне [0, 1] (раздел 2.6.1 диплома)."""
    print(f"🚀 Обучение нормализатора на {n_samples} примерах...")

    loader = CommonVoiceRULoader()
    pipeline = AudioPipeline()

    # Берём всех спикеров и случайно выбираем несколько
    all_speakers = loader.load_speakers()
    sample_speakers = np.random.choice(all_speakers, size=min(n_samples, len(all_speakers)), replace=False)

    features_list = []
    processed = 0

    for speaker_id in sample_speakers:
        phrases = loader.load_phrases_for_speaker(speaker_id, max_phrases=5)
        for phrase in phrases:
            audio_path = phrase["audio_path"]
            try:
                features, _ = pipeline.extract_features_detailed(audio_path)
                features_list.append(features)
                processed += 1
                if processed % 50 == 0:
                    print(f"  → обработано {processed}/{n_samples} записей")
                if processed >= n_samples:
                    break
            except Exception as e:
                print(f"  ⚠️ Пропущен {audio_path}: {e}")
        if processed >= n_samples:
            break

    if not features_list:
        raise ValueError("Не удалось обработать ни одной записи!")

    pipeline.normalizer.fit(features_list)
    print(f"\n✅ Нормализатор успешно обучен на {len(features_list)} примерах!")
    print(f"   Файл создан: models/audio_params/normalizer_params.json")
    print("   Теперь вектор всегда нормализован к [0, 1] — готов к Главе 3 (НПБК)")

if __name__ == "__main__":
    fit_normalizer(n_samples=1000)   # можно поставить 500–1000 для большей точности