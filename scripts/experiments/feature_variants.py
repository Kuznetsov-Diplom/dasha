# scripts/experiments/feature_variants.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random

from scripts.cv_ru_loader import CommonVoiceRULoader
from scripts.pipeline import AudioPipeline

def run_feature_experiment(num_runs: int = 8, n_own_phrases: int = 8):
    print(f"🚀 Эксперимент: {num_runs} итераций, {n_own_phrases} фраз\n")

    loader = CommonVoiceRULoader()
    pipeline = AudioPipeline()
    all_speakers = loader.load_speakers(min_phrases=10)

    all_results = []

    for run in range(1, num_runs + 1):
        own_speaker = random.choice(all_speakers)
        print(f"\n{'='*90}")
        print(f"ИТЕРАЦИЯ {run}/{num_runs} | Свой: {own_speaker[:12]}...")
        print(f"{'='*90}")

        own_phrases = loader.load_phrases_for_speaker(own_speaker, max_phrases=30)[:n_own_phrases]
        own_vectors = []
        for p in own_phrases:
            try:
                vec, _ = pipeline.extract_features_detailed(p["audio_path"])
                own_vectors.append(vec)
                print(f"  ✓ Успешно ({vec.shape[0]} признаков)")
            except Exception as e:
                print(f"  ✗ Ошибка: {e}")
        own_vectors = np.array(own_vectors)

        if len(own_vectors) < 2:
            print("  ⚠️ Мало векторов — пропуск итерации")
            continue

        # Чужие спикеры
        other_speakers = [s for s in all_speakers if s != own_speaker]
        num_foreign = random.randint(12, 18)
        selected_foreign = random.sample(other_speakers, min(num_foreign, len(other_speakers)))

        foreign_vectors = []
        for foreign_id in selected_foreign:
            phrases = loader.load_phrases_for_speaker(foreign_id, max_phrases=n_own_phrases)
            for p in phrases:
                try:
                    vec, _ = pipeline.extract_features_detailed(p["audio_path"])
                    foreign_vectors.append(vec)
                except:
                    continue
        foreign_vectors = np.array(foreign_vectors) if foreign_vectors else np.empty((0, own_vectors.shape[1]))

        def calc_corr(own, foreign):
            if len(own) < 2 or len(foreign) == 0:
                return 0.0, 0.0, 0.0
            intra = float(np.mean(cosine_similarity(own)[np.triu_indices_from(cosine_similarity(own), k=1)]))
            inter = float(np.mean(cosine_similarity(own, foreign)))
            return intra, inter, intra - inter

        # Варианты (работают с любой размерностью)
        dim = own_vectors.shape[1]
        variants = [
            (f"Полные ({dim})", own_vectors, foreign_vectors),
            ("Только Mean", own_vectors[:, :dim//2], foreign_vectors[:, :dim//2]),
            ("Только Std", own_vectors[:, dim//2:], foreign_vectors[:, dim//2:]),
            ("Первые 13", own_vectors[:, :13], foreign_vectors[:, :13]),
            ("Первые 20", own_vectors[:, :20], foreign_vectors[:, :20]),
            ("Первые 26", own_vectors[:, :26], foreign_vectors[:, :26]),
        ]

        for name, own_v, foreign_v in variants:
            intra, inter, diff = calc_corr(own_v, foreign_v)
            all_results.append({
                "Итерация": run,
                "Вариант": name,
                "Intra (Свой)": intra,
                "Inter (Чужие)": inter,
                "Разница": diff,
                "Кол-во чужих": len(selected_foreign)
            })

        # Таблица текущей итерации
        df_iter = pd.DataFrame([r for r in all_results if r["Итерация"] == run])
        print(df_iter[["Вариант", "Intra (Свой)", "Inter (Чужие)", "Разница"]].round(4))

    df_all = pd.DataFrame(all_results)
    if df_all.empty:
        print("❌ Не удалось собрать ни одной валидной итерации")
        return

    summary = df_all.groupby("Вариант").mean(numeric_only=True).round(4)

    print("\n" + "="*100)
    print("📊 СВОДНАЯ ТАБЛИЦА (средние)")
    print("="*100)
    print(summary[["Intra (Свой)", "Inter (Чужие)", "Разница"]])

    best = summary["Разница"].idxmax()
    print(f"\n🏆 Лучший вариант: {best} (разница {summary.loc[best, 'Разница']:.4f})")

if __name__ == "__main__":
    run_feature_experiment(num_runs=8, n_own_phrases=8)