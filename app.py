import gradio as gr
import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from PIL import Image
import io
import matplotlib.pyplot as plt

# ==================== Dummy 26-мерный вектор (пока) ====================
def extract_26_features(audio: np.ndarray, sr: int = 16000) -> list[float]:
    """Заглушка. Скоро заменим на настоящий AudioPreprocessor по Главе 2"""
    # 13 mean + 13 std
    mfcc_mean = np.random.randn(13).tolist()
    mfcc_std = np.random.randn(13).tolist()
    return mfcc_mean + mfcc_std


# ==================== Загрузка русских данных ====================
@st.cache_resource(show_spinner=True)
def load_russian_data():
    print("📥 Загружаем русский Common Voice (один спикер)...")
    ds = load_dataset("mozilla-foundation/common_voice_17_0", "ru", streaming=True, split="train")
    samples = []
    seen_client = None
    for ex in ds:
        if len(samples) >= 12:  # достаточно для демо
            break
        client_id = ex.get("client_id")
        if client_id and (seen_client is None or client_id == seen_client):
            if len(ex["audio"]["array"]) > 8000:  # отбрасываем очень короткие
                samples.append(ex)
                if seen_client is None:
                    seen_client = client_id
    print(f"✅ Загружено {len(samples)} фраз от русского спикера")
    return samples


samples = load_russian_data()


# ==================== Основная обработка ====================
def process_audio(selected_idx: int, uploaded_audio=None):
    try:
        if uploaded_audio is not None:
            sr, audio_np = uploaded_audio
            text = "Загруженный файл"
        else:
            ex = samples[selected_idx % len(samples)]
            audio_np = ex["audio"]["array"]
            sr = ex["audio"]["sampling_rate"]
            text = ex["sentence"][:100]

        # Resample → 16 кГц
        if sr != 16000:
            waveform = torch.from_numpy(audio_np).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_np = resampler(waveform).squeeze().numpy()

        features = extract_26_features(audio_np)

        # ==================== Визуализации ====================
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        axs[0].plot(audio_np)
        axs[0].set_title("Waveform")
        axs[1].specgram(audio_np, Fs=16000)
        axs[1].set_title("Spectrogram")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        # КЛЮЧЕВОЙ ФИКС
        plot_img = Image.open(buf).convert("RGB")

        return (
            text,
            (16000, audio_np.astype(np.float32)),
            plot_img,           # PIL.Image
            plot_img,           # MFCC plot (пока тот же)
            features,
            "✅ Успешно! 26-мерный вектор получен"
        )

    except Exception as e:
        err_msg = f"❌ Ошибка: {str(e)}"
        return err_msg, None, None, None, None, err_msg


# ==================== Gradio UI ====================
with gr.Blocks(title="Dasha — Глава 2") as demo:
    gr.Markdown("# 🎙️ Dasha — Тестовая версия (Глава 2)\n**Предварительная обработка речевого сигнала + 26-мерный вектор**")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Выберите трек русского спикера")
            track_dropdown = gr.Dropdown(
                choices=[f"{i}: {s['sentence'][:70]}..." for i, s in enumerate(samples)],
                label="Трек",
                value=0
            )
            listen_btn = gr.Button("▶ Прослушать выбранный трек", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Или загрузите свой файл")
            upload_audio = gr.Audio(
                label="Загрузить аудио",
                type="numpy",
                sources=["upload", "microphone"]
            )

    process_btn = gr.Button("🔬 Обработать по Главе 2", variant="primary", size="large")

    with gr.Row():
        with gr.Column():
            text_out = gr.Textbox(label="Текст фразы")
            audio_out = gr.Audio(label="Воспроизведение")
            features_out = gr.JSON(label="26-мерный вектор признаков")
            status_out = gr.Textbox(label="Статус")

        with gr.Column():
            plot_out = gr.Image(label="Waveform + Spectrogram")

    # Прослушка
    listen_btn.click(
        lambda idx: (16000, samples[int(idx) % len(samples)]["audio"]["array"]),
        inputs=track_dropdown,
        outputs=audio_out
    )

    # Обработка
    process_btn.click(
        process_audio,
        inputs=[track_dropdown, upload_audio],
        outputs=[text_out, audio_out, plot_out, plot_out, features_out, status_out]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),          # теперь здесь, а не в Blocks
        share=False
    )
