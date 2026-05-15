import os
import gradio as gr
from dotenv import load_dotenv
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from io import BytesIO
import base64

load_dotenv()

MIN_PHRASES_PER_SPEAKER = int(os.getenv('MIN_PHRASES_PER_SPEAKER', 10))

# Placeholder imports - replace with actual modules once created
# from cv_ru_loader import load_speakers_and_phrases
# from pipeline import process_phrase
# from feature_normalizer import normalize_features

# For now - mock data to make app run
speakers = [
    {'speaker_id': 'speaker_001', 'name': 'Спикер 001', 'phrases': [{'id': 0, 'text': 'Привет, как дела?', 'audio_path': 'data/sample.wav'}]},
    {'speaker_id': 'speaker_002', 'name': 'Спикер 002', 'phrases': [{'id': 0, 'text': 'Тестовая фраза', 'audio_path': 'data/sample.wav'}]}
]

def get_speakers():
    return [s['name'] for s in speakers]

def get_phrases(speaker_name):
    speaker = next((s for s in speakers if s['name'] == speaker_name), None)
    if speaker:
        return [p['text'] for p in speaker['phrases']]
    return []

def play_audio(speaker_name, phrase_text):
    # Mock audio
    return 'data/sample.wav'  # Gradio will handle local path

def process_phrase(speaker_name, phrase_text):
    steps = [
        "1. Загрузка аудио фразы",
        "2. Предподчеркивание (Pre-emphasis)",
        "3. VAD - определение речевых сегментов",
        "4. Выделение MFCC признаков",
        "5. Нормализация в диапазон [0, 1] (по ГОСТ)",
        "6. Формирование вектора признаков",
        "7. НПБК - преобразование в биометрический код (скелет)",
        "✅ Обработка завершена успешно!"
    ]
    
    # Mock results
    result_text = "\n".join(steps)
    
    # Mock plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(np.random.randn(100))
    ax.set_title('Пример waveform')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plot_html = f'<img src="data:image/png;base64,{plot_base64}">' 
    
    return result_text, plot_html

# Gradio Interface
with gr.Blocks(title="Dasha - Биометрическая обработка речи", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 Dasha — Система биометрической обработки речи
    
    **Одно окно для демонстрации полного пайплайна**
    Выберите спикера → фразу → обработайте по алгоритму ГОСТ
    """")
    
    with gr.Row():
        with gr.Column(scale=1):
            speaker_dropdown = gr.Dropdown(
                label="Выберите спикера",
                choices=get_speakers(),
                value=get_speakers()[0] if get_speakers() else None
            )
            
            phrase_dropdown = gr.Dropdown(
                label="Выберите фразу",
                choices=[],
                value=None
            )
            
            audio_player = gr.Audio(
                label="Прослушать аудио",
                type="filepath",
                interactive=False
            )
            
            process_btn = gr.Button("🚀 Обработать фразу", variant="primary", size="large")
        
        with gr.Column(scale=2):
            steps_output = gr.Markdown(label="📋 Ход обработки (по алгоритму)")
            visualization = gr.HTML(label="Визуализация этапов")
    
    def update_phrases(speaker_name):
        return gr.update(choices=get_phrases(speaker_name))
    
    speaker_dropdown.change(
        fn=update_phrases,
        inputs=speaker_dropdown,
        outputs=phrase_dropdown
    )
    
    def on_play(speaker, phrase):
        audio_path = play_audio(speaker, phrase)
        return audio_path
    
    phrase_dropdown.change(
        fn=on_play,
        inputs=[speaker_dropdown, phrase_dropdown],
        outputs=audio_player
    )
    
    process_btn.click(
        fn=process_phrase,
        inputs=[speaker_dropdown, phrase_dropdown],
        outputs=[steps_output, visualization]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
