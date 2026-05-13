# app.py
import gradio as gr
from scripts.cv_ru_loader import CommonVoiceRULoader
from scripts.pipeline import AudioPipeline

loader = CommonVoiceRULoader()
pipeline = AudioPipeline()

# speaker_id → {text: audio_path}
speaker_phrase_map: dict[str, dict[str, str]] = {}

def load_all_speakers():
    speakers = loader.load_speakers()
    return gr.update(choices=speakers, value=speakers[0] if speakers else None)

def get_phrases_for_speaker(speaker_id: str):
    if speaker_id not in speaker_phrase_map:
        raw_phrases = loader.load_phrases_for_speaker(speaker_id, max_phrases=20)
        speaker_phrase_map[speaker_id] = {p["text"]: p["audio_path"] for p in raw_phrases}
    
    texts = list(speaker_phrase_map[speaker_id].keys())
    first_text = texts[0] if texts else None
    return gr.update(choices=texts, value=first_text)

def on_phrase_change(speaker_id: str, selected_text: str | None):
    if not selected_text or speaker_id not in speaker_phrase_map:
        return "", None, gr.Audio(visible=False)
    
    audio_path = speaker_phrase_map[speaker_id][selected_text]
    return (
        selected_text,
        audio_path,
        gr.Audio(value=audio_path, autoplay=True, visible=True)
    )

def process_audio(audio_input):
    """Универсальная обработка: принимает и filepath, и (sr, array) от Gradio."""
    if isinstance(audio_input, tuple):          # Gradio numpy-режим
        sr, y = audio_input
        # временно сохраняем в память (или можно сохранить на диск, но для демо хватит)
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, y, sr)
            audio_path = f.name
    else:                                       # filepath
        audio_path = audio_input

    if not audio_path:
        return None, "❌ Выберите аудиофайл"
    
    features, meta = AudioPipeline.extract_features(audio_path)
    info = f"✅ 26-мерный вектор извлечён\nАктивных фреймов: {meta['n_active_frames']}"
    return features.tolist(), info

with gr.Blocks(title="Dasha — Биометрия по голосу (Глава 2)") as demo:
    gr.Markdown("# Dasha — генерация криптографических ключей по голосу\n**Глава 2**: предварительная обработка + 26-мерный вектор признаков")

    with gr.Tabs():
        with gr.TabItem("📁 Из датасета (по умолчанию)"):
            with gr.Row():
                speaker_dropdown = gr.Dropdown(label="Спикер (client_id)", choices=[], interactive=True, scale=2)
                phrase_dropdown = gr.Dropdown(label="Фраза", choices=[], interactive=True, scale=3)

            audio_player = gr.Audio(
                label="Прослушать фразу",
                interactive=False,
                type="filepath"          # ← КРИТИЧНЫЙ ФИКС
            )
            text_display = gr.Textbox(label="Текст фразы", interactive=False)

            gr.Markdown("---")
            btn_extract = gr.Button(
                "🔬 Извлечь 26-мерный вектор биометрических признаков",
                variant="primary",
                size="large"
            )

            with gr.Row():
                output_features = gr.JSON(label="26-мерный вектор (13 mean + 13 std)")
                output_info = gr.Textbox(label="Информация")

        with gr.TabItem("📤 Загрузить свой файл"):
            upload = gr.Audio(label="Загрузите свой аудиофайл (WAV/MP3)", type="filepath")
            btn_extract_upload = gr.Button("🔬 Извлечь 26-мерный вектор", variant="primary", size="large")
            with gr.Row():
                output_features_upload = gr.JSON(label="26-мерный вектор")
                output_info_upload = gr.Textbox(label="Информация")

    # Логика
    speaker_dropdown.change(get_phrases_for_speaker, speaker_dropdown, phrase_dropdown)
    phrase_dropdown.change(on_phrase_change, [speaker_dropdown, phrase_dropdown], [text_display, audio_player, audio_player])
    btn_extract.click(process_audio, audio_player, [output_features, output_info])
    btn_extract_upload.click(process_audio, upload, [output_features_upload, output_info_upload])

    demo.load(load_all_speakers, outputs=speaker_dropdown)

demo.launch(theme=gr.themes.Base(), share=False)