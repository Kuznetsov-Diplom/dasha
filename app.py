import gradio as gr
import os
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

from cv_ru_loader import load_speakers_and_phrases, get_random_speaker
from pipeline import process_voice, extract_mfcc_features
from feature_normalizer import FeatureNormalizer
from npbk import NPBKConverter

# Загрузка данных
speakers = load_speakers_and_phrases()

npbk = NPBKConverter()

def get_random_speaker():
    if not speakers:
        return None
    return random.choice(list(speakers.keys()))

def process_voice(speaker_id, phrase_text):
    if speaker_id not in speakers:
        return None, "Спикер не найден"
    speaker_data = speakers[speaker_id]
    # Находим фразу
    for item in speaker_data:
        if item['text'] == phrase_text:
            audio_path = item['audio_path']
            steps = process_audio_file(audio_path)
            waveform = steps['original']
            pre_emphasis = steps['pre_emphasis']
            mfcc_plot = steps.get('mfcc_plot', np.zeros(13))
            normalized_vector = steps['normalized_vector']
            return waveform, pre_emphasis, mfcc_plot, normalized_vector, "Обработка завершена успешно!"
    return None, "Фраза не найдена"

def plot_normalized_vector(vec):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(vec)), vec)
    ax.set_title('Нормализованный вектор признаков [0, 1]')
    ax.set_xlabel('Индекс признака')
    ax.set_ylabel('Значение')
    ax.grid(True)
    return fig

def correlation_own(speaker_id, num_phrases):
    if speaker_id not in speakers:
        return None, "Спикер не найден"
    speaker_data = speakers[speaker_id]
    if len(speaker_data) < num_phrases:
        num_phrases = len(speaker_data)
    selected = speaker_data[:num_phrases]
    vectors = []
    for item in selected:
        steps = process_audio_file(item['audio_path'])
        vec = steps['normalized_vector']
        vectors.append(vec)
    corr_matrix = np.corrcoef(vectors)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='viridis')
    ax.set_title('Корреляция векторов "Свои"')
    plt.colorbar(im)
    return fig, f"Корреляция рассчитана для {num_phrases} фраз спикера {speaker_id}"

# Основной интерфейс
with gr.Blocks(title="Dasha — Система биометрической обработки речи", theme=gr.themes.Dark()) as demo:
    gr.Markdown("# Dasha — Система биометрической обработки речи")
    gr.Markdown("### Одно окно • Выберите спикера + фразу → обработка по алгоритму ГОСТ")
    
    with gr.Tabs():
        # Вкладка 1
        with gr.Tab("1. Обработка голоса"):
            with gr.Row():
                gr.Button("🎲 Выбрать случайного спикера", variant="primary").click(
                    lambda: gr.update(value=get_random_speaker()), inputs=None, outputs=gr.Dropdown()
                )
            speaker_dropdown = gr.Dropdown(choices=list(speakers.keys()), label="Спикер", value=list(speakers.keys())[0] if speakers else None)
            
            phrase_dropdown = gr.Dropdown(label="Фраза", choices=[], interactive=True)
            
            audio_player = gr.Audio(label="Прослушать аудио", type="numpy")
            
            process_btn = gr.Button("🔬 Обработать фразу", variant="primary", size="large")
            
            status = gr.Markdown()
            
            with gr.Row():
                with gr.Column():
                
                    gr.Markdown("**1. Оригинальный waveform**")
                    orig_wave = gr.Plot()
                with gr.Column():
                    gr.Markdown("**2. После Pre-emphasis**")
                    pre_wave = gr.Plot()
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**3. MFCC**")
                    mfcc_plot = gr.Plot()
                with gr.Column():
                    gr.Markdown("**4. Нормализованный вектор [0,1]**")
                    norm_plot = gr.Plot()
        
        # Вкладка 2
        with gr.Tab("2. Обнаружение корреляции среди своих"):
            with gr.Row():
                gr.Button("🎲 Случайный спикер", variant="primary").click(
                    lambda: gr.update(value=get_random_speaker()), inputs=None, outputs=gr.Dropdown()
                )
            speaker2 = gr.Dropdown(choices=list(speakers.keys()), label="Спикер", value=list(speakers.keys())[0] if speakers else None)
            num_phrases_slider = gr.Slider(minimum=2, maximum=20, value=5, step=1, label="Количество фраз для анализа")
            corr_btn = gr.Button("📊 Вычислить корреляцию", variant="primary")
            corr_status = gr.Markdown()
            corr_heatmap = gr.Plot()
        
        # Остальные вкладки (пока пустые)
        with gr.Tab("3. Обнаружение корреляции с чужими"):
            gr.Markdown("### Скоро: сравнение \"Свой\" vs \"Чужой\"")
        with gr.Tab("4. Регистрация данных в НБК"):
            gr.Markdown("### Регистрация спикера в Нейросетевом Преобразователе Биометрия-Код")
            reg_btn = gr.Button("Зарегистрировать спикера в НБК")
        with gr.Tab("5. Восстановление ключа в НПБК"):
            gr.Markdown("### Восстановление криптографического ключа")
            key_btn = gr.Button("Сгенерировать / Восстановить ключ")
        with gr.Tab("6. Корреляция ответов НПБК (ошибки 1 и 2 рода)"):
            gr.Markdown("### Анализ FAR / FRR и корреляций выходов НПБК")
        with gr.Tab("7. Эксперименты"):
            gr.Markdown("### Эксперименты, тесты производительности, длина ключа и т.д.")
    
    # Логика обновления списка фраз
    def update_phrases(speaker_id):
        if speaker_id and speaker_id in speakers:
            phrases = [item['text'] for item in speakers[speaker_id]]
            return gr.update(choices=phrases, value=phrases[0] if phrases else None)
        return gr.update(choices=[])
    
    speaker_dropdown.change(update_phrases, inputs=speaker_dropdown, outputs=phrase_dropdown)
    
    # Обработка голоса
    def on_process(speaker_id, phrase):
        waveform, pre_emphasis, mfcc_p, norm_vec, msg = process_voice(speaker_id, phrase)
        if waveform is None:
            return None, None, None, None, msg
        orig_fig = plt.figure()
        plt.plot(waveform[0])
        plt.title("Оригинальный waveform")
        pre_fig = plt.figure()
        plt.plot(pre_emphasis[0])
        plt.title("После Pre-emphasis")
        norm_fig = plot_normalized_vector(norm_vec)
        return orig_fig, pre_fig, None, norm_fig, msg
    
    process_btn.click(
        on_process,
        inputs=[speaker_dropdown, phrase_dropdown],
        outputs=[orig_wave, pre_wave, mfcc_plot, norm_plot, status]
    )
    
    # Корреляция среди своих
    corr_btn.click(
        correlation_own,
        inputs=[speaker2, num_phrases_slider],
        outputs=[corr_heatmap, corr_status]
    )

if __name__ == "__main__":
    demo.launch()
