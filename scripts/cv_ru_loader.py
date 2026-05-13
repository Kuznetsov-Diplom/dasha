# scripts/cv_ru_loader.py
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

class CommonVoiceRULoader:
    """Загрузчик Common Voice Russian (firefox-ru-dataset).
    Полностью соответствует требованиям Главы 2."""

    def __init__(self, dataset_root: str = "data/firefox-ru-dataset"):
        self.base_dir = Path(dataset_root).resolve()
        self.clips_dir = self.base_dir / "clips"
        self.validated_path = self.base_dir / "validated.tsv"
        self.speaker_col = "client_id"

        if not self.validated_path.exists():
            raise FileNotFoundError(f"TSV не найден: {self.validated_path}")

        print(f"✅ CommonVoiceRULoader инициализирован → {self.base_dir}")

    def load_speakers(self, split: str = "crowd_train") -> List[str]:
        """Возвращает список client_id."""
        df = pd.read_csv(self.validated_path, sep="\t", low_memory=False)
        speakers = sorted(df[self.speaker_col].unique().tolist())
        print(f"Загружено {len(speakers):,} спикеров")
        return speakers

    def _find_text_column(self, meta: pd.DataFrame) -> str:
        for col in ["sentence", "text", "transcription"]:
            if col in meta.columns:
                return col
        return "sentence"

    def load_phrases_for_speaker(self, speaker_id: str, split: str = "crowd_train", max_phrases: int = 20) -> List[Dict[str, Any]]:
        """Возвращает фразы спикера (совместимо с app.py)."""
        df = pd.read_csv(self.validated_path, sep="\t", low_memory=False)
        df_speaker = df[df[self.speaker_col] == speaker_id].head(max_phrases).copy()

        phrases = []
        text_col = self._find_text_column(df)
        for idx, row in df_speaker.iterrows():
            phrases.append({
                "phrase_idx": idx,
                "text": row.get(text_col, ""),
                "audio_path": str(self.clips_dir / str(row["path"])),
            })
        return phrases