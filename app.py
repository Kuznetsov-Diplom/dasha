import gradio as gr
import numpy as np
import plotly.graph_objects as go
import random
import os
from pipeline import process_phrase
from cv_ru_loader import load_speakers_and_phrases

speakers = load_speakers_and_phrases()

N_FEATURES = 39

def get_audio_info(audio_path):
    import soundfile as sf
    try:
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1: y = y[:, 0]
        return y, sr
    except:
        return np.zeros(16000), 16000

def process_one_phrase(mode, speaker, phrase_idx, file):
    if mode == "Датасет":
        if not speaker or phrase_idx is None:
            return "Выберите спикера и фразу", None, None, None, None
        path = speakers[speaker]['phrases'][int(phrase_idx)]['audio_path']
        label = f"{speaker} — {speakers[speaker]['phrases'][int(phrase_idx)]['sentence'][:50]}..."
    else:
        if file is None:
            return "Загрузите файл", None, None, None, None
        path = file.name
        label = os.path.basename(file.name)

    result = process_phrase(path)
    y, sr = get_audio_info(path)

    # 1. Waveform
    t = np.linspace(0, len(y)/sr, len(y))
    fig_wave = go.Figure(go.Scatter(x=t, y=y, mode='lines'))
    fig_wave.update_layout(title="1. Исходный сигнал", height=280)

    # 2. Нормализованный вектор (39-dim)
    fig_vec = go.Figure(go.Bar(x=list(range(N_FEATURES)), y=result['normalized_vector']))
    fig_vec.update_layout(title=f"2. Нормализованный вектор ({N_FEATURES}-dim)", height=320, yaxis_range=[0, 1])

    # 3. MFCC
    if 'mfcc' in result and result['mfcc'] is not None:
        fig_mfcc = go.Figure(go.Heatmap(z=result['mfcc'].T, colorscale='Viridis'))
        fig_mfcc.update_layout(title="3. MFCC спектрограмма", height=320)
    else:
        fig_mfcc = go.Figure()
        fig_mfcc.update_layout(title="MFCC не доступен")

    return (f"### ✅ {label}", fig_wave, fig_vec, fig_mfcc, None)

with gr.Blocks(title="Dasha — Система биометрической обработки речи", theme=gr.themes.Base()) as demo:
    gr.Markdown("# Dasha — Система биометрической обработки речи")
    with gr.Tabs():
        with gr.TabItem("1. Пошаговая обработка одной фразы"):
            gr.Markdown("## Визуализация обработки одной записи")
            with gr.Row():
                with gr.Column(scale=1):
                    mode1 = gr.Radio(["Датасет", "Мои файлы"], value="Датасет", label="Источник")
                    with gr.Group() as ds1:
                        sp1 = gr.Dropdown(list(speakers.keys()), label="Спикер")
                        random_btn1 = gr.Button("🎲 Случайный спикер", size="sm")
                        ph1 = gr.Dropdown([], label="Фраза")
                    with gr.Group(visible=False) as fl1:
                        f1 = gr.File(label="Загрузите аудиофайл", file_types=[".wav", ".mp3"])
                    btn1 = gr.Button("🚀 Обработать", variant="primary", size="lg")
                with gr.Column(scale=2):
                    out_label = gr.Markdown()
                    with gr.Accordion("1. Исходный сигнал", open=True): p1 = gr.Plot()
                    with gr.Accordion("2. Нормализованный вектор", open=True): p2 = gr.Plot()
                    with gr.Accordion("3. MFCC спектрограмма", open=False): p3 = gr.Plot()
            mode1.change(lambda m: (gr.update(visible=m=="Датасет"), gr.update(visible=m=="Мои файлы")), inputs=mode1, outputs=[ds1, fl1])
            random_btn1.click(lambda: random.choice(list(speakers.keys())), outputs=sp1)
            sp1.change(lambda s: gr.update(choices=[(p['sentence'][:60], str(i)) for i, p in enumerate(speakers.get(s, {}).get('phrases', []))] if s in speakers else []), inputs=sp1, outputs=ph1)
            btn1.click(process_one_phrase, [mode1, sp1, ph1, f1], [out_label, p1, p2, p3])

        with gr.TabItem("2. Корреляция среди своих записей"):
            gr.Markdown("## Анализ нескольких записей")
            # (оставим упрощённо для начала)
            gr.Markdown("В разработке...")

        with gr.TabItem("3. Массовый тест по ГОСТ"):
            gr.Markdown("## Массовый тест (в разработке)")
            gr.Markdown("В разработке...")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)