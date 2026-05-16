# fit_normalizer.py
from cv_ru_loader import load_speakers_and_phrases
from pipeline import AudioPipeline
import numpy as np
from feature_normalizer import FeatureNormalizer

def fit_normalizer(max_speakers=100, max_phrases_per_speaker=20):
    print("🚀 Запуск обучения MinMax-нормализатора [0,1]...")
    loader = load_speakers_and_phrases  # функция из cv_ru_loader
    speakers = loader()  # уже с фильтром по MIN_PHRASES_PER_SPEAKER

    pipeline = AudioPipeline()
    features_list = []

    processed = 0
    for speaker_id, info in list(speakers.items())[:max_speakers]:
        for phrase in info["phrases"][:max_phrases_per_speaker]:
            try:
                raw_26 = pipeline.extract_raw_26(phrase["audio_path"])
                features_list.append(raw_26)
                processed += 1
                if processed % 50 == 0:
                    print(f"   Обработано {processed} фраз...")
            except Exception as e:
                print(f"⚠️ Пропущена фраза {phrase['audio_path']}: {e}")
                continue

    if not features_list:
        print("❌ Не удалось извлечь ни одного вектора")
        return

    normalizer = FeatureNormalizer()
    normalizer.fit(features_list)
    print(f"🎉 Нормализация завершена! Обработано {len(features_list)} векторов.")

if __name__ == "__main__":
    fit_normalizer(max_speakers=500, max_phrases_per_speaker=15)  # можно увеличить