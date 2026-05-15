import gradio as gr
from pathlib import Path
from cv_ru_loader import load_speakers_and_phrases
from pipeline import process_phrase

def create_interface():
    speakers_data = load_speakers_and_phrases()

    speaker_choices = []
    speaker_to_phrases = {}
    for speaker_id, info in speakers_data.items():
        short_id = speaker_id[-8:] if len(speaker_id) > 8 else speaker_id
        count = len(info["phrases"])
        display_name = f"{short_id} ({count} фраз)"
        speaker_choices.append(display_name)
        speaker_to_phrases[display_name] = (speaker_id, info["phrases"])

    def update_phrases(speaker_display):
        if not speaker_display:
            return gr.update(choices=[]), None
        speaker_id, phrases = speaker_to_phrases[speaker_display]
        phrase_choices = [p["sentence"] for p in phrases]
        audio_path = phrases[0]["audio_path"] if phrases else None
        return gr.update(choices=phrase_choices, value=phrase_choices[0] if phrase_choices else None), audio_path

    def get_audio_path(speaker_display, phrase_text):
        if not speaker_display or not phrase_text:
            return None
        speaker_id, phrases = speaker_to_phrases[speaker_display]
        for p in phrases:
            if p["sentence"] == phrase_text:
                return p["audio_path"]
        return None

    def process_and_show(speaker_display, phrase_text):
        audio_path = get_audio_path(speaker_display, phrase_text)
        if not audio_path:
            return [None] * 4, "❌ Ошибка: файл не найден"

        try:
            result = process_phrase(audio_path)
            
            vector_text = "\n".join([f"{i+1:2d}. {v:.6f}" for i, v in enumerate(result["normalized_vector"])])
            
            return (
                result["waveform_plot"],
                result["pre_emphasis_plot"],
                result["mfcc_plot"],
                f"✅ Нормализованный вектор [0, 1] (26 значений):\n\n{vector_text}",
                "✅ Обработка завершена успешно!"
            )
        except Exception as e:
            return [None] * 4, f"❌ Ошибка обработки: {str(e)}"

    with gr.Blocks(title="Dasha — Биометрия речи", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Dasha — Система биометрической обработки речи")
        gr.Markdown("**Одно окно** • Выберите спикера + фразу → обработка по алгоритму ГОСТ")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Выберите спикера")
                speaker_dropdown = gr.Dropdown(
                    choices=speaker_choices,
                    label="Спикер",
                    value=speaker_choices[0] if speaker_choices else None
                )
                
                gr.Markdown("### Выберите фразу")
                phrase_dropdown = gr.Dropdown(label="Фраза", choices=[], interactive=True)
                
                audio_player = gr.Audio(label="🎧 Прослушать аудио", interactive=False, type="filepath")
                
                process_btn = gr.Button("🚀 Обработать фразу", variant="primary", size="large")

            with gr.Column(scale=2):
                gr.Markdown("### Результаты обработки (по ГОСТ)")
                
                status = gr.Markdown("Нажмите кнопку «Обработать фразу»")

                with gr.Row():
                    with gr.Column():
                        waveform_plot = gr.Plot(label="1. Оригинальный waveform")
                        pre_plot = gr.Plot(label="2. После Pre-emphasis")
                    
                    with gr.Column():
                        mfcc_plot = gr.Plot(label="3. MFCC")
                        vector_output = gr.Textbox(label="4. Нормализованный вектор [0, 1]", lines=15, show_copy_button=True)

        # События
        speaker_dropdown.change(
            fn=update_phrases,
            inputs=speaker_dropdown,
            outputs=[phrase_dropdown, audio_player]
        )

        phrase_dropdown.change(
            fn=get_audio_path,
            inputs=[speaker_dropdown, phrase_dropdown],
            outputs=audio_player
        )

        process_btn.click(
            fn=process_and_show,
            inputs=[speaker_dropdown, phrase_dropdown],
            outputs=[waveform_plot, pre_plot, mfcc_plot, vector_output, status]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()