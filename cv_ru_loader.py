# scripts/cv_ru_loader.py
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

class CommonVoiceRULoader:
    """Временная версия — только 10 спикеров для отладки."""

    def __init__(self, dataset_root: str = "data/firefox-ru-dataset"):
        self.base_dir = Path(dataset_root).resolve()
        self.clips_dir = self.base_dir / "clips"
        self.validated_path = self.base_dir / "validated.tsv"
        self.speaker_col = "client_id"

    def load_speakers(self) -> List[str]:
        df = pd.read_csv(self.validated_path, sep="\t", low_memory=False)
        speakers = sorted(df[self.speaker_col].unique().tolist())
        limited = speakers[:10000]                    # ← только 10 спикеров
        print(f"✅ DEBUG: Загружено {len(limited)} спикеров (ограничено для отладки)")
        return limited

    def load_phrases_for_speaker(self, speaker_id: str, max_phrases: int = 20) -> List[Dict[str, Any]]:
        df = pd.read_csv(self.validated_path, sep="\t", low_memory=False)
        df_speaker = df[df[self.speaker_col] == speaker_id].head(max_phrases)

        phrases = []
        text_col = "sentence" if "sentence" in df.columns else "text"
        for _, row in df_speaker.iterrows():
            phrases.append({
                "text": str(row.get(text_col, "")),
                "audio_path": str(self.clips_dir / str(row["path"])),
            })
        return phrases
