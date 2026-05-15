# cv_ru_loader.py
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

MIN_PHRASES_PER_SPEAKER = int(os.getenv("MIN_PHRASES_PER_SPEAKER", 10))

class CommonVoiceRULoader:
    """Загрузчик датасета Common Voice RU.
    Поддерживает фильтрацию по минимальному количеству фраз (из .env).
    """

    def __init__(self, dataset_root: str = "data/firefox-ru-dataset"):
        self.base_dir = Path(dataset_root).resolve()
        self.clips_dir = self.base_dir / "clips"
        self.validated_path = self.base_dir / "validated.tsv"
        self.speaker_col = "client_id"

    def load_speakers(self) -> List[str]:
        """Возвращает список всех client_id."""
        if not self.validated_path.exists():
            raise FileNotFoundError(f"Не найден validated.tsv в {self.validated_path.parent}")
        df = pd.read_csv(self.validated_path, sep="\t", low_memory=False)
        speakers = sorted(df[self.speaker_col].unique().tolist())
        print(f"✅ Загружено {len(speakers)} спикеров из датасета")
        return speakers

    def load_phrases_for_speaker(self, speaker_id: str, max_phrases: int = 20) -> List[Dict[str, Any]]:
        """Возвращает список фраз для спикера."""
        df = pd.read_csv(self.validated_path, sep="\t", low_memory=False)
        df_speaker = df[df[self.speaker_col] == speaker_id].head(max_phrases)

        phrases = []
        text_col = "sentence" if "sentence" in df.columns else "text"
        for _, row in df_speaker.iterrows():
            phrases.append({
                "text": str(row.get(text_col, "").strip()),
                "audio_path": str(self.clips_dir / str(row["path"])),
            })
        return phrases


def load_speakers_and_phrases() -> Dict[str, List[Dict[str, Any]]]:
    """Главный вход для UI: возвращает dict {speaker_id: [phrase_dicts]} 
    Только спикеры с MIN_PHRASES_PER_SPEAKER фразами.
    """
    loader = CommonVoiceRULoader()
    all_speakers = loader.load_speakers()
    
    result = {}
    for speaker_id in all_speakers:
        phrases = loader.load_phrases_for_speaker(speaker_id, max_phrases=50)  # берём с запасом
        if len(phrases) >= MIN_PHRASES_PER_SPEAKER:
            # приводим ключи к тому, что ожидает ui.py
            converted_phrases = [{
                "sentence": p["text"],
                "path": p["audio_path"]
            } for p in phrases[:20]]  # ограничиваем 20 фразами для UI
            result[speaker_id] = converted_phrases
    
    print(f"✅ load_speakers_and_phrases: найдено {len(result)} спикеров с >= {MIN_PHRASES_PER_SPEAKER} фразами")
    return result

if __name__ == "__main__":
    data = load_speakers_and_phrases()
    print(f"Тест: {len(data)} спикеров загружено")
