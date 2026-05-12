import gradio as gr
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt   # ← добавили
from pathlib import Path
import pandas as pd
from PIL import Image
from io import BytesIO
import os

from dasha.audio.pipeline import pipeline

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   # ← добавил os


# ====================== ЗАГРУЗКА ДАННЫХ ======================
def load_russian_data(max_samples=15):
    print("📥 Загружаем Dusha crowd_train...")
    wav_dir = Path("data/dusha/crowd_train/wavs")
    tsv_path = Path("data/dusha/crowd_train/raw_crowd_train.tsv")

    meta = pd.read_csv(tsv_path, sep='\t')
    speaker_col = 'hash_id'
    top_speaker = meta[speaker_col].value_counts().index[0]

    speaker_meta = meta[meta[speaker_col] == top_speaker].head(max_samples*2)

    samples = []
    for _, row in speaker_meta.iterrows():
        filename = Path(str(row.get('audio_path', ''))).name
        full_path = wav_dir / filename
        if full_path.exists():
            try:
                audio_np, sr = sf.read(full_path, dtype='float32')
                if len(audio_np.shape) > 1:
                    audio_np = np.mean(audio_np, axis=1)

                text = str(row.get('speaker_text') or f"Запись {filename[:12]}...").strip()
                samples.append({
                    "audio": {"array": audio_np, "sampling_rate": int(sr)},
                    "sentence": text[:250],
                    "path": str(full_path)
                })
                if len(samples) >= max_samples:
                    break
            except:
                continue

    print(f"✅ Загружено {len(samples)} записей спикера {top_speaker}")
    return samples


samples = load_russian_data()


def create_plot(audio_np: np.ndarray, sr: int) -> Image.Image:
    """Waveform + Spectrogram"""
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    axs[0].plot(audio_np[:min(len(audio_np), sr*12)])
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


def process_audio(selected_idx: str, uploaded=None):
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
        idx = int(selected_idx)
        ex = samples[idx % len(samples)]
        sentence = ex["sentence"]
        features, meta = pipeline.extract_features(ex["path"])
        audio_np = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]

    plot_pil = create_plot(audio_np, sr)
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

    return (
        sentence,
        (sr, audio_int16),
        plot_pil,
        features.tolist(),
        f"✅ {meta['duration_sec']} сек | Активно: {meta.get('active_ratio', 0)*100:.1f}%"
    )


# ====================== ИНТЕРФЕЙС ======================
with gr.Blocks(title="Dasha — Глава 2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Dasha — Глава 2: Предобработка + Признаки (по ГОСТ Р 52633)")

    # Выбор записи (закреплено сверху)
    with gr.Row():
        with gr.Column(scale=3):
            track_dropdown = gr.Dropdown(
                choices=[(f"{i}: {s['sentence'][:75]}...", str(i)) for i, s in enumerate(samples)],
                label="🎤 Выберите фразу спикера",
                value="0",
                interactive=True
            )
        with gr.Column(scale=1):
            listen_btn = gr.Button("▶ Прослушать", variant="primary", size="large")

    gr.Markdown("---")

    # Основной пайплайн
    with gr.Row():
        with gr.Column(scale=1):
            upload_audio = gr.Audio(
                label="📤 Загрузить свой WAV (16 кГц)",
                type="numpy",
                sources=["upload", "microphone"]
            )
            process_btn = gr.Button("🔬 Запустить полный pipeline Главы 2", 
                                  variant="primary", size="large")

        with gr.Column(scale=2):
            text_out = gr.Textbox(label="📝 Прочитанный текст", lines=2)
            audio_out = gr.Audio(label="🎧 Воспроизведение")
            status_out = gr.Textbox(label="📊 Статус обработки")

    gr.Markdown("### 📈 1. Исходный сигнал")
    plot_out = gr.Image(label="Waveform + Spectrogram", height=550)

    gr.Markdown("### 🔢 2. 26-мерный вектор биометрических признаков")
    features_out = gr.JSON(label="Вектор (13 mean + 13 std)")

    # ====================== События ======================
    def on_track_change(idx: str):
        ex = samples[int(idx) % len(samples)]
        return (
            ex["sentence"],
            (ex["audio"]["sampling_rate"], ex["audio"]["array"]),
            ex["sentence"]
        )

    track_dropdown.change(
        on_track_change,
        inputs=track_dropdown,
        outputs=[text_out, audio_out, text_out]
    )

    listen_btn.click(
        lambda idx: (samples[int(idx) % len(samples)]["audio"]["sampling_rate"],
                     samples[int(idx) % len(samples)]["audio"]["array"]),
        inputs=track_dropdown,
        outputs=audio_out
    )

    process_btn.click(
        process_audio,
        inputs=[track_dropdown, upload_audio],
        outputs=[text_out, audio_out, plot_out, features_out, status_out]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()      # теперь правильно
    )