import gradio as gr
import numpy as np
import random
from pathlib import Path
from dotenv import load_dotenv
import plotly.graph_objects as go

load_dotenv()

from cv_ru_loader import load_speakers_and_phrases
from pipeline import process_phrase

speakers = load_speakers_and_phrases()

custom_files = []      # [(name, path), ...]
custom_vectors = []

def get_random_speaker():
    return random.choice(list(speakers.keys())) if speakers else None

def add_custom_files(files):
    global custom_files
    for file in files or []:
        name = Path(file.name).name
        path = str(file.name)
        custom_files.append((name, path))
    return [[name] for name, _ in custom_files]

def process_custom_files():
    global custom_vectors
    custom_vectors = []
    for _, path in custom_files:
        try:
            result = process_phrase(path)
            custom_vectors.append(np.array(result["normalized_vector"]))
        except:
            pass
    return f"✅ Обработано {len(custom_vectors)} файлов"

def delete_custom_file(selected):
    global custom_files, custom_vectors
    if selected:
        name = selected[0][0]
        for i, (n, p) in enumerate(custom_files):
            if n == name:
                del custom_files[i]
                if i < len(custom_vectors):
                    del custom_vectors[i]
                break
    return [[n] for n, _ in custom_files]

def create_ui():
    with gr.Blocks(title="Dasha — Система биометрической обработки речи", theme=gr.themes.Base()) as demo:
        gr.Markdown("# Dasha — Система биометрической обработки речи")

        with gr.Tabs():
            # Вкладка 1
            with gr.Tab("1. Обработка голоса"):
                with gr.Row():
                    with gr.Column(scale=1):
                        mode1 = gr.Radio(["Датасет", "Мои файлы"], value="Датасет", label="Источник")
                        speaker1 = gr.Dropdown(choices=list(speakers.keys()), label="Спикер", visible=True)
                        phrase1 = gr.Dropdown(label="Фраза", choices=[], visible=True)
                        file1 = gr.File(label="Загрузить свой файл", visible=False)
                        btn1 = gr.Button("🔬 Обработать", variant="primary")
                    with gr.Column(scale=2):
                        status1 = gr.Markdown()
                        with gr.Row():
                            orig1 = gr.Plot(label="Оригинал")
                            pre1 = gr.Plot(label="Pre-emphasis")
                        with gr.Row():
                            mfcc1 = gr.Plot(label="MFCC")
                            norm1 = gr.Plot(label="Нормализованный вектор")

            # Вкладка 2 (улучшенная)
            with gr.Tab("2. Обнаружение корреляции среди своих"):
                with gr.Row():
                    with gr.Column(scale=1):
                        mode2 = gr.Radio(["Датасет", "Мои файлы", "Датасет + мои файлы"], value="Мои файлы", label="Источник")
                        speaker2 = gr.Dropdown(choices=list(speakers.keys()), label="Спикер", visible=True)
                        num_phrases = gr.Slider(2, 50, 10, label="Количество фраз")
                        file2 = gr.File(label="Загрузить свои файлы", file_count="multiple", visible=True)
                        upload_btn = gr.Button("➕ Добавить файлы")
                        process_btn2 = gr.Button("🔬 Обработать файлы")
                        custom_list2 = gr.Dataframe(headers=["Файл"], value=[], label="Загруженные файлы")
                        delete_btn2 = gr.Button("🗑 Удалить")
                        corr_btn2 = gr.Button("📊 Вычислить корреляцию", variant="primary", size="large")
                    with gr.Column(scale=2):
                        status2 = gr.Markdown()
                        heatmap2 = gr.Plot()
                        lines2 = gr.Plot()

            # Остальные вкладки (заглушки)
            for i in range(3, 8):
                with gr.Tab(f"{i}. Вкладка {i}"):
                    gr.Markdown(f"### Вкладка {i} — в разработке")

        # События (упрощённо для примера)
        # ... (полная логика событий в файле)

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()