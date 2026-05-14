# fit_normalizer.py
import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scripts.cv_ru_loader import CommonVoiceRULoader
from scripts.pipeline import AudioPipeline

def fit_normalizer(n_samples: int = 1000):
    """Обучаем FeatureNormalizer (Z-score) на реальных записях из CommonVoice RU.
    Использует extract_raw_26 — строго по требованиям Главы 2 (сохранение mean/std).
    """
    print(f"🚀 Обучение Z-нормализатора на {n_samples} примерах (Глава 2)...")

    # 🔥 Удаляем старый MinMax-файл, если он остался
    params_path = Path("models/audio_params/normalizer_params.json")
    if params_path.exists():
        print("🗑️  Старый normalizer_params.json (MinMax) удалён")
        params_path.unlink()

    loader = CommonVoiceRULoader()
    pipeline = AudioPipeline()

    all_speakers = loader.load_speakers()
    sample_speakers = np.random.choice(all_speakers, size=min(n_samples, len(all_speakers)), replace=False)

    features_list: list[np.ndarray] = []
    processed = 0

    for speaker_id in sample_speakers:
        phrases = loader.load_phrases_for_speaker(speaker_id, max_phrases=5)
        for phrase in phrases:
            audio_path = phrase["audio_path"]
            try:
                raw_26 = pipeline.extract_raw_26(audio_path)  # СЫРОЙ вектор!
                features_list.append(raw_26)
                processed += 1
                if processed % 100 == 0:
                    print(f"  → обработано {processed}/{n_samples} записей")
                if processed >= n_samples:
                    break
            except Exception as e:
                print(f"  ⚠️  Пропущен {audio_path}: {e}")
        if processed >= n_samples:
            break

    if not features_list:
        raise ValueError("Не удалось обработать ни одной записи!")

    pipeline.normalizer.fit(features_list)  # fit на СЫРЫХ векторах!
    print(f"\n✅ Z-нормализатор успешно обучен на {len(features_list)} примерах!")
    print(f"   Параметры сохранены: models/audio_params/normalizer_params.json")
    print("   Теперь 26-мерный вектор готов к Главе 3 (нейросетевой преобразователь по ГОСТ Р 52633)!")

if __name__ == "__main__":
    fit_normalizer(n_samples=1000)
