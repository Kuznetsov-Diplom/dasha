# app.py
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from scripts.cv_ru_loader import CommonVoiceRULoader
from scripts.pipeline import AudioPipeline

loader = CommonVoiceRULoader()
pipeline = AudioPipeline()

speaker_phrase_map: dict[str, dict[str, str]] = {}

def load_all_speakers():
    speakers = loader.load_speakers()
    return gr.update(choices=speakers, value=speakers[0] if speakers else None)

def get_phrases_for_speaker(speaker_id: str):
    if speaker_id not in speaker_phrase_map:
        raw = loader.load_phrases_for_speaker(speaker_id, max_phrases=20)
        speaker_phrase_map[speaker_id] = {p["text"]: p["audio_path"] for p in raw}
    texts = list(speaker_phrase_map[speaker_id].keys())
    return gr.update(choices=texts, value=texts[0] if texts else None)

def on_phrase_change(speaker_id: str, selected_text: str | None):
    if not selected_text or speaker_id not in speaker_phrase_map:
        return "", None, gr.Audio(visible=False)
    audio_path = speaker_phrase_map[speaker_id][selected_text]
    return selected_text, audio_path, gr.Audio(value=audio_path, autoplay=True, visible=True)

def process_audio(audio_input):
    """Основной пайплайн (Глава 2) — один файл."""
    if isinstance(audio_input, tuple):
        sr, y = audio_input
        import tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, y, sr)
            audio_path = f.name
    else:
        audio_path = audio_input

    if not audio_path:
        return None, "❌ Выберите файл", None, None, None, None

    features, data = pipeline.extract_features_detailed(audio_path)
    steps = data["steps"]
    meta = data["meta"]

    # Waveform + Pre-emphasis
    fig_wave, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(steps["original_waveform"]); ax[0].set_title("Оригинальный waveform")
    ax[1].plot(steps["pre_emphasis_waveform"]); ax[1].set_title("После Pre-emphasis (α=0.97)")
    plt.tight_layout()

    # Energy + VAD
    fig_vad, ax = plt.subplots(figsize=(10, 3))
    ax.plot(steps["energy"], label="Energy")
    ax.plot(steps["vad_mask"] * steps["energy"].max(), label="VAD mask", alpha=0.7)
    ax.set_title("Energy + Voice Activity Detection")
    ax.legend()
    plt.tight_layout()

    # MFCC Heatmap
    fig_mfcc, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(steps["mfcc"], aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("MFCC Heatmap (13 коэффициентов)")
    fig_mfcc.colorbar(im, ax=ax)
    plt.tight_layout()

    # 26-вектор
    fig_vec, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].bar(range(26), steps["raw_26_vector"])
    ax[0].set_title("Raw 26-мерный вектор")
    ax[1].bar(range(26), steps["normalized_26_vector"])
    ax[1].set_title("Normalized [0, 1] (для НПБК)")
    plt.tight_layout()

    info = f"✅ 26-мерный вектор готов (нормализован [0,1])\nАктивных фреймов: {meta['n_active_frames']}"

    return features.tolist(), info, fig_wave, fig_vad, fig_mfcc, fig_vec

def check_speaker_stability(speaker_id: str, n_phrases: int):
    """Проверка стабильности признаков одного спикера (intra-speaker correlation)."""
    if not speaker_id:
        return None, "❌ Выберите спикера", None

    phrases = loader.load_phrases_for_speaker(speaker_id, max_phrases=50)
    if len(phrases) < n_phrases:
        n_phrases = len(phrases)

    selected = np.random.choice(phrases, size=n_phrases, replace=False)

    vectors = []
    for p in selected:
        features, _ = pipeline.extract_features_detailed(p["audio_path"])
        vectors.append(features)

    vectors = np.array(vectors)  # (N, 26)

    # Косинусная корреляция
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(vectors)

    mean_sim = float(np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)]))
    status = "✅ Стабильность высокая" if mean_sim > 0.85 else "⚠️ Стабильность средняя (проверьте запись)"

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"Матрица косинусной корреляции ({n_phrases} фраз одного спикера)")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    info = f"""Спикер: {speaker_id[:12]}...
Обработано фраз: {n_phrases}
Средняя корреляция: {mean_sim:.4f}
{status}"""

    return sim_matrix.tolist(), info, fig

with gr.Blocks(title="Dasha — Глава 2") as demo:
    gr.Markdown("# Dasha — генерация криптографических ключей по голосу\n**Глава 2**: предварительная обработка + 26-мерный вектор")

    with gr.Tabs():
        # === ВКЛАДКА 1: Основной пайплайн ===
        with gr.TabItem("📁 Из датасета (по умолчанию)"):
            # ... (весь предыдущий код вкладки без изменений) ...
            with gr.Row():
                speaker_dropdown = gr.Dropdown(label="Спикер (client_id)", choices=[], interactive=True, scale=2)
                phrase_dropdown = gr.Dropdown(label="Фраза", choices=[], interactive=True, scale=3)

            audio_player = gr.Audio(label="Прослушать фразу", interactive=False, type="filepath")
            text_display = gr.Textbox(label="Текст фразы", interactive=False)

            gr.Markdown("---")
            btn_extract = gr.Button("🔬 Запустить полный пайплайн Главы 2", variant="primary", size="large")

            output_features = gr.JSON(label="26-мерный вектор [0,1]")
            output_info = gr.Textbox(label="Результаты обработки")

            with gr.Accordion("📊 Пошаговая визуализация обработки (Глава 2)", open=True):
                with gr.Row():
                    plot_wave = gr.Plot(label="Waveform + Pre-emphasis")
                    plot_vad = gr.Plot(label="Energy + VAD")
                with gr.Row():
                    plot_mfcc = gr.Plot(label="MFCC Heatmap")
                    plot_vector = gr.Plot(label="26-мерный вектор (raw → normalized)")

        # === НОВАЯ ВКЛАДКА: ПРОВЕРКА СТАБИЛЬНОСТИ ===
        with gr.TabItem("🔍 Анализ стабильности спикера"):
            gr.Markdown("**Проверка intra-speaker корреляции** — насколько похожи векторы одного спикера (должна быть высокой)")
            speaker_stab = gr.Dropdown(label="Выберите спикера", choices=[], interactive=True)
            n_slider = gr.Slider(3, 15, value=8, step=1, label="Количество случайных фраз")
            btn_stab = gr.Button("🔬 Проверить корреляцию", variant="primary", size="large")

            stab_info = gr.Textbox(label="Результат анализа")
            stab_heatmap = gr.Plot(label="Heatmap корреляции векторов")

    # Логика основной вкладки (без изменений)
    speaker_dropdown.change(get_phrases_for_speaker, speaker_dropdown, phrase_dropdown)
    phrase_dropdown.change(on_phrase_change, [speaker_dropdown, phrase_dropdown], [text_display, audio_player, audio_player])
    btn_extract.click(process_audio, audio_player, [output_features, output_info, plot_wave, plot_vad, plot_mfcc, plot_vector])

    # Логика новой вкладки
    speaker_stab.change(lambda x: x, speaker_stab, speaker_stab)  # просто для обновления
    btn_stab.click(check_speaker_stability, [speaker_stab, n_slider], [stab_heatmap, stab_info, stab_heatmap])

    # Загрузка спикеров при старте (для обеих вкладок)
    demo.load(load_all_speakers, outputs=[speaker_dropdown, speaker_stab])

demo.launch(share=False)