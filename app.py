import gradio as gr
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from io import BytesIO
import os
import pandas as pd

from dasha.audio.pipeline import pipeline
from dasha.data.dusha_loader import DushaLoader

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ====================== ЗАГРУЗКА ДАННЫХ ======================
loader = DushaLoader()

speaker_phrases: dict[str, list[dict]] = {}
all_speakers: list[str] = []
speaker_stats: dict[str, int] = {}
current_split = "crowd_train"

def load_all_speakers(split: str = "crowd_train"):
    global all_speakers, speaker_stats, current_split
    current_split = split
    print(f"🔄 Загружаем спикеров из {split}...")
    all_speakers = loader.load_speakers(split=split)
    
    tsv_path = loader.base_dir / split / f"raw_{split}.tsv"
    meta = pd.read_csv(tsv_path, sep="\t")
    text_col = loader._find_text_column(meta)
    stats = meta.groupby(loader.speaker_col)[text_col].nunique()
    speaker_stats = stats.to_dict()
    
    total = sum(speaker_stats.values())
    avg = total / len(speaker_stats) if speaker_stats else 0
    print(f"✅ {len(all_speakers):,} спикеров | уникальных фраз: {total:,} | среднее: {avg:.1f}")
    return all_speakers

load_all_speakers(split="crowd_train")

def get_phrases_for_speaker(speaker_id: str, max_phrases: int = 20):
    if speaker_id in speaker_phrases:
        return speaker_phrases[speaker_id]
    phrases = loader.load_phrases_for_speaker(speaker_id, split=current_split, max_phrases=max_phrases)
    speaker_phrases[speaker_id] = phrases
    return phrases

def create_plot(audio_np: np.ndarray, sr: int) -> Image.Image:
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    axs[0].plot(audio_np[:min(len(audio_np), sr * 12)])
    axs[0].set_title("Waveform")
    axs[0].set_xlabel("Samples")
    axs[0].grid(True, alpha=0.3)

    axs[1].specgram(audio_np, Fs=sr, NFFT=1024, noverlap=512, cmap='viridis')
    axs[1].set_title("Spectrogram")
    axs[1].set_xlabel("Time (s)")

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def process_audio(selected_audio_path: str | None, uploaded=None):
    if uploaded is not None and len(uploaded) == 2 and uploaded[1] is not None:
        sr, array = uploaded
        audio_np = np.array(array, dtype=np.float32)
        sr = int(sr)
        sentence = "🔊 Загруженный пользователем файл"
        temp_path = Path("temp_upload.wav")
        sf.write(temp_path, audio_np, sr)
        features, meta = pipeline.extract_features(temp_path)
        temp_path.unlink(missing_ok=True)
    else:
        if not selected_audio_path:
            return "❌ Нет аудио", None, None, None, "Ошибка"
        audio_path = Path(selected_audio_path)
        sentence = "Запись из датасета Dusha"
        features, meta = pipeline.extract_features(audio_path)
        audio_np, sr = sf.read(str(audio_path), dtype='float32')
        if len(audio_np.shape) > 1:
            audio_np = np.mean(audio_np, axis=1)

    plot_pil = create_plot(audio_np, sr)
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

    return (
        sentence,
        (sr, audio_int16),
        plot_pil,
        features.tolist(),
        f"✅ {meta['duration_sec']:.2f} сек | Активно: {meta.get('active_ratio', 0)*100:.1f}%"
    )

# ====================== ГРАДИО ======================
with gr.Blocks(title="Dasha — Глава 2") as demo:
    gr.Markdown("# 🎙️ Dasha — Глава 2: Предобработка + Признаки (по ГОСТ Р 52633)")

    with gr.Row():
        split_dropdown = gr.Dropdown(choices=["crowd_train", "crowd_test"], value="crowd_train",
                                    label="📂 Сплит датасета", interactive=True)
        with gr.Column(scale=3):
            top_speakers = sorted(speaker_stats.items(), key=lambda x: x[1], reverse=True)[:500]
            speaker_choices = [(f"{spk} ({cnt} уникальных фраз)", spk) for spk, cnt in top_speakers]
            speaker_dropdown = gr.Dropdown(choices=speaker_choices,
                                          value=speaker_choices[0][1] if speaker_choices else None,
                                          label="👤 Спикер", interactive=True)
        with gr.Column(scale=1):
            reload_btn = gr.Button("🔄 Перезагрузить", variant="secondary")

    phrase_dropdown = gr.Dropdown(choices=[], label="📝 Фраза (текст + путь)", interactive=True)

    gr.Markdown("---")

    with gr.Row():
        upload_audio = gr.Audio(label="📤 Или загрузить свой WAV / микрофон (16 кГц)",
                               type="numpy", sources=["upload", "microphone"])
        process_btn = gr.Button("🚀 Запустить pipeline Главы 2", variant="primary", size="large")

    with gr.Row():
        text_out = gr.Textbox(label="Текст записи", lines=2)
        status_out = gr.Textbox(label="Статус обработки")

    audio_out = gr.Audio(label="🎧 Воспроизведение")

    gr.Markdown("### 1. Исходный сигнал")
    plot_out = gr.Image(label="Waveform + Spectrogram", height=550)

    gr.Markdown("### 2. 26-мерный вектор биометрических признаков")
    features_out = gr.JSON(label="MFCC (13 mean + 13 std)")

    # ====================== СОБЫТИЯ ======================
    def update_phrases(selected_speaker_id: str):
        if not selected_speaker_id:
            return gr.update(choices=[], value=None)
        phrases = get_phrases_for_speaker(selected_speaker_id)
        if not phrases:
            return gr.update(choices=[("Нет фраз", None)], value=None)
        
        # Делаем value уникальным, чтобы Gradio всегда реагировал на смену фразы
        choices = [(f"{p['phrase_idx']}: {p['text']}", f"{p['audio_path']}||{p['phrase_idx']}") for p in phrases]
        return gr.update(choices=choices, value=choices[0][1] if choices else None)

    speaker_dropdown.change(update_phrases, inputs=speaker_dropdown, outputs=phrase_dropdown)

    def on_phrase_change(selected_value: str):
        if not selected_value or "||" not in selected_value:
            return "❌ Нет пути", None, None, None, "Ошибка"
        audio_path = selected_value.split("||")[0]   # отрезаем индекс
        return process_audio(audio_path)

    phrase_dropdown.change(
        on_phrase_change,
        inputs=phrase_dropdown,
        outputs=[text_out, audio_out, plot_out, features_out, status_out]
    )

    def change_split(new_split: str):
        load_all_speakers(split=new_split)
        top_speakers = sorted(speaker_stats.items(), key=lambda x: x[1], reverse=True)[:500]
        speaker_choices = [(f"{spk} ({cnt} уникальных фраз)", spk) for spk, cnt in top_speakers]
        return gr.update(choices=speaker_choices, value=speaker_choices[0][1] if speaker_choices else None)

    split_dropdown.change(change_split, inputs=split_dropdown, outputs=speaker_dropdown)

    def reload_speakers():
        load_all_speakers(split=current_split)
        top_speakers = sorted(speaker_stats.items(), key=lambda x: x[1], reverse=True)[:500]
        speaker_choices = [(f"{spk} ({cnt} уникальных фраз)", spk) for spk, cnt in top_speakers]
        return gr.update(choices=speaker_choices, value=speaker_choices[0][1] if speaker_choices else None)

    reload_btn.click(reload_speakers, outputs=speaker_dropdown)

    process_btn.click(
        process_audio,
        inputs=[phrase_dropdown, upload_audio],
        outputs=[text_out, audio_out, plot_out, features_out, status_out]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )