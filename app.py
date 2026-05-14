# app.py
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from scripts.cv_ru_loader import CommonVoiceRULoader
from scripts.pipeline import AudioPipeline

loader = CommonVoiceRULoader()
pipeline = AudioPipeline()

speaker_phrase_map: dict = {}

def load_all_speakers():
    speakers = loader.load_speakers()
    print(f"✅ Загружено {len(speakers)} спикеров")
    update = gr.update(choices=speakers, value=speakers[0] if speakers else None)
    return update, update

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
    meta = data["meta"]

    fig_wave, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(data["steps"]["original_waveform"])
    ax[0].set_title("Original Waveform")
    ax[1].plot(data["steps"]["pre_emphasis_waveform"])
    ax[1].set_title("After Pre-emphasis")
    plt.tight_layout()

    fig_vad, ax = plt.subplots(figsize=(10, 3))
    ax.plot(data["steps"]["energy"], label="Energy")
    ax.plot(data["steps"]["vad_mask"] * data["steps"]["energy"].max(), label="VAD", alpha=0.7)
    ax.legend()
    ax.set_title("Energy + VAD")
    plt.tight_layout()

    fig_mfcc, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data["steps"]["mfcc"], aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("MFCC Heatmap")
    fig_mfcc.colorbar(im, ax=ax)
    plt.tight_layout()

    fig_vec, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].bar(range(26), data["steps"]["raw_26_vector"])
    ax[0].set_title("Raw 26-vector")
    ax[1].bar(range(26), data["steps"]["normalized_26_vector"])
    ax[1].set_title("Normalized [0,1]")
    plt.tight_layout()

    info = f"✅ 26-мерный вектор готов\nАктивных фреймов: {meta.get('n_active_frames', 0)}"
    return features.tolist(), info, fig_wave, fig_vad, fig_mfcc, fig_vec


def check_speaker_stability(speaker_id: str, n_phrases: int):
    """Свой (N фраз) vs 16+ Чужих (по N фраз от каждого)"""
    # Свои фразы
    phrases_own = loader.load_phrases_for_speaker(speaker_id, max_phrases=30)
    n_phrases = min(n_phrases, len(phrases_own))
    selected_own = np.random.choice(phrases_own, size=n_phrases, replace=False)

    vectors_own = []
    for p in selected_own:
        try:
            vec, _ = pipeline.extract_features_detailed(p["audio_path"])
            vectors_own.append(vec)
        except:
            continue
    vectors_own = np.array(vectors_own)

    # 16 случайных чужих спикеров
    all_speakers = loader.load_speakers()
    other_speakers = [s for s in all_speakers if s != speaker_id]
    num_foreign = min(16, len(other_speakers))
    selected_foreign_ids = np.random.choice(other_speakers, size=num_foreign, replace=False)

    vectors_foreign = []
    for foreign_id in selected_foreign_ids:
        phrases_f = loader.load_phrases_for_speaker(foreign_id, max_phrases=n_phrases)
        sel_f = np.random.choice(phrases_f, size=min(n_phrases, len(phrases_f)), replace=False)
        for p in sel_f:
            try:
                vec, _ = pipeline.extract_features_detailed(p["audio_path"])
                vectors_foreign.append(vec)
            except:
                continue

    vectors_foreign = np.array(vectors_foreign) if vectors_foreign else np.empty((0, 26))

    # Корреляции
    from sklearn.metrics.pairwise import cosine_similarity
    sim_own = cosine_similarity(vectors_own) if len(vectors_own) > 1 else np.array([[1.0]])
    mean_intra = float(np.mean(sim_own[np.triu_indices_from(sim_own, k=1)])) if len(vectors_own) > 1 else 1.0

    mean_inter = 0.0
    if len(vectors_foreign) > 0:
        sim_inter = cosine_similarity(vectors_own, vectors_foreign)
        mean_inter = float(np.mean(sim_inter))

    # Визуализация
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Overlay своих фраз (синий)
    for vec in vectors_own:
        axs[0, 0].plot(range(26), vec, alpha=0.6, color="blue", label="Свой")
    axs[0, 0].set_title("Overlay векторов СВОИХ фраз")
    axs[0, 0].legend()

    # Overlay чужих (красный)
    for vec in vectors_foreign[:16]:  # максимум 16 для читаемости
        axs[0, 1].plot(range(26), vec, alpha=0.3, color="red", label="Чужой")
    axs[0, 1].set_title("Overlay векторов ЧУЖИХ фраз")
    axs[0, 1].legend()

    # Heatmap
    all_vectors = np.vstack([vectors_own, vectors_foreign]) if len(vectors_foreign) > 0 else vectors_own
    sim_all = cosine_similarity(all_vectors)
    im = axs[1, 0].imshow(sim_all, cmap="viridis", vmin=0, vmax=1)
    axs[1, 0].set_title("Heatmap всех векторов (Свой + Чужие)")
    fig.colorbar(im, ax=axs[1, 0])

    # Сравнение intra vs inter
    axs[1, 1].bar(["Intra (Свой)", "Inter (Чужие)"], [mean_intra, mean_inter], color=["green", "red"])
    axs[1, 1].set_title("Средняя корреляция")
    axs[1, 1].set_ylim(0, 1)
    plt.tight_layout()

    info = f"""Свой спикер: {speaker_id[:12]}...
Фраз: {n_phrases}
Intra-correlation: {mean_intra:.4f} {'✅ Высокая' if mean_intra > 0.85 else '⚠️ Средняя'}
Inter-correlation (16+ чужих): {mean_inter:.4f} {'✅ Низкая' if mean_inter < 0.6 else '⚠️ Высокая'}"""

    return fig, info, fig


with gr.Blocks(title="Dasha — Глава 2") as demo:
    gr.Markdown("# Dasha — Глава 2: предварительная обработка + 26-мерный вектор")

    with gr.Tabs():
        with gr.TabItem("📁 Из датасета"):
            with gr.Row():
                speaker_dropdown = gr.Dropdown(label="Спикер (client_id)", choices=[], interactive=True)
                phrase_dropdown = gr.Dropdown(label="Фраза", choices=[], interactive=True)

            audio_player = gr.Audio(label="Прослушать", type="filepath")
            text_display = gr.Textbox(label="Текст фразы", interactive=False)

            btn_extract = gr.Button("🔬 Запустить пайплайн Главы 2", variant="primary", size="large")

            output_features = gr.JSON(label="26-мерный вектор [0,1]")
            output_info = gr.Textbox(label="Результат")

            with gr.Accordion("📊 Визуализация обработки", open=True):
                with gr.Row():
                    plot_wave = gr.Plot()
                    plot_vad = gr.Plot()
                with gr.Row():
                    plot_mfcc = gr.Plot()
                    plot_vector = gr.Plot()

        with gr.TabItem("🔍 Стабильность + Свой vs Чужие (16+)"):
            speaker_stab = gr.Dropdown(label="Свой спикер", choices=[], interactive=True)
            n_slider = gr.Slider(2, 15, value=6, label="Кол-во фраз (для Своего и каждого Чужого)")
            btn_stab = gr.Button("Проверить стабильность (Свой vs 16+ Чужих)", variant="primary", size="large")
            stab_info = gr.Textbox(label="Результат")
            stab_heatmap = gr.Plot()

    speaker_dropdown.change(get_phrases_for_speaker, speaker_dropdown, phrase_dropdown)
    phrase_dropdown.change(on_phrase_change, [speaker_dropdown, phrase_dropdown], [text_display, audio_player, audio_player])

    btn_extract.click(process_audio, audio_player, [output_features, output_info, plot_wave, plot_vad, plot_mfcc, plot_vector])

    btn_stab.click(check_speaker_stability, [speaker_stab, n_slider], [stab_heatmap, stab_info, stab_heatmap])

    demo.load(load_all_speakers, outputs=[speaker_dropdown, speaker_stab])

demo.launch(server_name="0.0.0.0", server_port=7860, share=False, quiet=True)
