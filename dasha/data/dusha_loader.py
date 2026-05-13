from pathlib import Path
from typing import Dict, List
import pandas as pd

class DushaLoader:
    """Модульный загрузчик датасета Dusha с улучшенной диагностикой путей (crowd_train + crowd_test)"""
    
    def __init__(self, base_dir: str | Path = "data/dusha"):
        self.base_dir = Path(base_dir)
        self.speaker_col = "hash_id"
        self.text_cols = ["speaker_text", "text", "transcription", "normalized_text"]
        self.path_cols = ["path", "audio_path", "wav_path", "audio_filename", "filename"]

    def _find_text_column(self, meta: pd.DataFrame) -> str:
        for col in self.text_cols:
            if col in meta.columns:
                print(f"✅ Найдена колонка текста: {col}")
                return col
        raise ValueError(f"Не найдена колонка текста! Доступные колонки: {list(meta.columns)}")

    def _find_path_column(self, meta: pd.DataFrame) -> str:
        print(f"🔍 Доступные колонки в TSV: {list(meta.columns)}")
        for col in self.path_cols:
            if col in meta.columns:
                print(f"✅ Найдена колонка с путём к аудио: {col}")
                return col
        print("⚠️ Не найдена стандартная колонка пути → используем fallback 'path'")
        return "path"

    def load_speakers(self, split: str = "crowd_test") -> List[str]:
        tsv_path = self.base_dir / split / f"raw_{split}.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(f"TSV не найден: {tsv_path}")
        meta = pd.read_csv(tsv_path, sep="\t")
        speakers = sorted(meta[self.speaker_col].unique().tolist())
        print(f"✅ Загружено {len(speakers)} спикеров из {split}")
        return speakers

    def load_phrases_for_speaker(
        self, speaker_id: str, split: str = "crowd_test", max_phrases: int = 20
    ) -> List[Dict]:
        """Загружает ТОЛЬКО УНИКАЛЬНЫЕ фразы + подробная диагностика путей"""
        tsv_path = self.base_dir / split / f"raw_{split}.tsv"
        wav_dir = self.base_dir / split / "wavs"
        
        meta = pd.read_csv(tsv_path, sep="\t")
        text_col = self._find_text_column(meta)
        path_col = self._find_path_column(meta)
        
        speaker_meta = meta[meta[self.speaker_col] == speaker_id].copy()
        
        seen_texts = set()
        phrases = []
        
        print(f"📂 wav_dir = {wav_dir}")
        print(f"🔍 Используем колонку пути: {path_col}")
        
        for idx, row in speaker_meta.iterrows():
            text = str(row.get(text_col) or "").strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            
            # Получаем путь к файлу
            raw_path = str(row.get(path_col, "") or row.get("path", "") or row.get("audio_path", ""))
            audio_filename = Path(raw_path).name
            full_audio_path = wav_dir / audio_filename
            
            exists = full_audio_path.exists()
            print(f"   → Фраза {len(phrases)}: '{text[:70]}...' | file: {audio_filename} → exists: {exists}")
            
            if exists and len(phrases) < max_phrases:
                phrases.append({
                    "speaker_id": speaker_id,
                    "phrase_idx": len(phrases),
                    "text": text[:130] + ("..." if len(text) > 130 else ""),
                    "audio_path": str(full_audio_path),
                    "audio_filename": audio_filename
                })
        
        print(f"✅ Для спикера {speaker_id} найдено {len(phrases)} уникальных фраз с аудио")
        if len(phrases) == 0:
            print("❌ ВНИМАНИЕ: Ни один аудиофайл не найден! Проверь структуру папки wavs/")
        
        return phrases