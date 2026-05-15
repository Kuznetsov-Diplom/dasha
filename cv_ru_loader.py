import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
from typing import Dict, List

load_dotenv()

class CommonVoiceRULoader:
    def __init__(self):
        self.data_dir = Path("data/firefox-ru-dataset")
        self.validated_path = self.data_dir / "validated.tsv"
        self.clips_dir = self.data_dir / "clips"
        self.min_phrases = int(os.getenv("MIN_PHRASES_PER_SPEAKER", 10))
        self._data = None

    def _load_data(self):
        """Загружаем validated.tsv ТОЛЬКО ОДИН раз"""
        if self._data is not None:
            return self._data
        if not self.validated_path.exists():
            raise FileNotFoundError(
                f"Файл не найден: {self.validated_path}\n"
                "Скачайте Common Voice RU датасет в папку data/firefox-ru-dataset/"
            )
        print(f"📂 Загружаем {self.validated_path.name} (это может занять 5–15 секунд)...")
        self._data = pd.read_csv(self.validated_path, sep="\t", low_memory=False)
        print(f"✅ Загружено {len(self._data):,} строк из validated.tsv")
        return self._data

    def load_speakers_and_phrases(self, max_speakers: int = 500) -> Dict[str, Dict]:
        """Быстрая загрузка всех спикеров"""
        df = self._load_data()
        
        # Группируем по client_id один раз
        grouped = df.groupby('client_id')
        
        speakers = {}
        count = 0
        
        for speaker_id, group in grouped:
            if len(group) < self.min_phrases:
                continue

            phrases = []
            for _, row in group.iterrows():
                audio_path = self.clips_dir / row['path']
                if audio_path.exists():
                    phrases.append({
                        "sentence": row['sentence'],
                        "audio_path": str(audio_path)
                    })
                if len(phrases) >= 50:          # ограничиваем для скорости
                    break

            if len(phrases) >= self.min_phrases:
                speakers[speaker_id] = {
                    "phrases": phrases,
                    "total_phrases": len(group)
                }
                count += 1
                if count >= max_speakers:
                    break

        print(f"✅ Загружено {len(speakers)} спикеров (минимум {self.min_phrases} фраз на спикера)")
        return speakers


# Для совместимости с ui.py
def load_speakers_and_phrases():
    loader = CommonVoiceRULoader()
    return loader.load_speakers_and_phrases()