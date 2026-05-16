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
print(f"✅ Загружено {len(speakers)} спикеров")

custom_files = []      # [(name, path), ...]
custom_vectors = []

def get_random_speaker():
    return random.choice(list(speakers.keys())) if speakers else None

def add_custom_files(files):
    global custom_files
    for f in files or []:
        name = Path(f.name).name
        path = str(f.name)
        custom_files.append((name, path))
    return [[n] for n, _ in custom_files]

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

def update_visibility(mode):
    '''Conditional visibility based on mode'''
    dataset_visible = mode in ["Датасет", "Датасет + мои файлы"]
    custom_visible = mode in ["Мои файлы", "Датасет + мои файлы"]
    return (
        gr.update(visible=dataset_visible),  # speaker2
        gr.update(visible=dataset_visible),  # num_phrases2
        gr.update(visible=custom_visible),  # file_uploader
        gr.update(visible=custom_visible),  # upload_btn
        gr.update(visible=custom_visible),  # process_custom_btn
        gr.update(visible=custom_visible),  # custom_list
        gr.update(visible=custom_visible),  # delete_btn
    )

def correlation_own(speaker_id, num_phrases, mode):
    global custom_vectors
    vectors = []
    labels = []

    # Датасет
    if mode in ["Датасет", "Датасет + мои файлы"]:
        if speaker_id in speakers:
            data = speakers[speaker_id]['phrases'][:int(num_phrases)]
            for item in data:
                try:
                    result = process_phrase(item['audio_path'])
                    vectors.append(np.array(result["normalized_vector"]))
                    labels.append(item['sentence'][:30] + "..." if len(item['sentence']) > 30 else item['sentence'])
                except:
                    continue

    # Свои файлы
    if mode in ["Мои файлы", "Датасет + мои файлы"]:
        for name, _ in custom_files:
            if len(vectors) < len(custom_vectors) + len(labels):
                idx = len(vectors) - len(labels)
                if idx < len(custom_vectors):
                    vectors.append(custom_vectors[idx])
                    labels.append(name)

    if len(vectors) < 2:
        return None, None, "❌ Нужно минимум 2 вектора"

    vectors = np.array(vectors)
    mean_vec = np.mean(vectors, axis=0)
    corr_matrix = np.corrcoef(vectors)

    corr_with_etalon = np.array([np.corrcoef(vectors[i], mean_vec)[0, 1] for i in range(len(vectors))])
    extended_matrix = np.hstack([corr_matrix, corr_with_etalon.reshape(-1, 1)])

    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
    avg_corr = float(np.mean(corr_matrix[mask])) if np.any(mask) else 0.0
    stability = "Отличная" if avg_corr > 0.85 else "Хорошая" if avg_corr > 0.75 else "Средняя" if avg_corr > 0.65 else "Низкая"

    labels_with_etalon = labels + ["Эталон"]
    fig_heat = go.Figure(data=go.Heatmap(
        z=extended_matrix,
        x=labels_with_etalon,
        y=labels,
        colorscale='RdYlBu_r',
        text=np.round(extended_matrix, 3),
        texttemplate="%{text}",
    ))
    fig_heat.update_layout(title=f'Корреляция + эталон ({len(vectors)} записей)', height=650)

    fig_lines = go.Figure()
    for i, vec in enumerate(vectors):
        fig_lines.add_trace(go.Scatter(x=list(range(26)), y=vec, mode='lines', name=labels[i], opacity=0.75))
    fig_lines.add_trace(go.Scatter(x=list(range(26)), y=mean_vec, mode='lines', name='Средний вектор (эталон)', line=dict(color='black', width=4)))
    fig_lines.update_layout(title="Векторы + средний вектор", height=500, legend_title="Файлы")

    status = f"✅ **Готово** ({len(vectors)} записей)<br>**Средняя корреляция:** {avg_corr:.3f}<br>**Стабильность:** {stability}"
    return fig_heat, fig_lines, status

# ==================== ИНТЕРФЕЙС ====================
with gr.Blocks(title="Dasha — Система биометрической обработки речи", theme=gr.themes.Base()) as demo:
    gr.Markdown("# Dasha — Система биометрической обработки речи")

    with gr.Tabs():
        # Вкладка 1
        with gr.Tab("1. Обработка голоса"):
            with gr.Row():
                with gr.Column(scale=1):
                    mode1 = gr.Radio(["Датасет", "Мои файлы"], value="Датасет", label="Источник")
                    speaker1 = gr.Dropdown(choices=list(speakers.keys()), label="Спикер")
                    phrase1 = gr.Dropdown(label="Фраза")
                    file1 = gr.File(label="Загрузить свой файл")
                    btn1 = gr.Button("🔬 Обработать", variant="primary")
                with gr.Column(scale=2):
                    status1 = gr.Markdown()
                    with gr.Row():
                        gr.Plot()
                        gr.Plot()
                    with gr.Row():
                        gr.Plot()
                        gr.Plot()

        # Вкладка 2
        with gr.Tab("2. Обнаружение корреляции среди своих"):
            with gr.Row():
                with gr.Column(scale=1):
                    mode2 = gr.Radio(["Датасет", "Мои файлы", "Датасет + мои файлы"], value="Мои файлы", label="Что анализировать")
                    speaker2 = gr.Dropdown(choices=list(speakers.keys()), label="Спикер из датасета", visible=True)
                    num_phrases2 = gr.Slider(2, 50, 10, step=1, label="Количество фраз из датасета", visible=True)
                    
                    gr.Markdown("### Загрузить свои файлы")
                    file_uploader = gr.File(file_count="multiple", label="Выберите аудиофайлы", visible=True)
                    upload_btn = gr.Button("➕ Добавить файлы", variant="secondary", visible=True)
                    process_custom_btn = gr.Button("🔬 Обработать загруженные файлы", variant="primary", visible=True)
                    
                    custom_list = gr.Dataframe(headers=["Файл"], value=[], label="Загруженные файлы", visible=True)
                    delete_btn = gr.Button("🗑 Удалить выбранный файл", visible=True)
                    
                    corr_btn = gr.Button("📊 Вычислить корреляцию", variant="primary", size="large")

                with gr.Column(scale=2):
                    corr_status = gr.Markdown()
                    corr_heatmap = gr.Plot()
                    lines_plot = gr.Plot()

        # Остальные вкладки
        for i in range(3, 8):
            with gr.Tab(f"{i}. Вкладка {i}"):
                gr.Markdown(f"### Вкладка {i} — в разработке")

    # События
    upload_btn.click(add_custom_files, inputs=file_uploader, outputs=custom_list)
    process_custom_btn.click(process_custom_files, outputs=corr_status)
    delete_btn.click(delete_custom_file, inputs=custom_list, outputs=custom_list)
    corr_btn.click(
        correlation_own,
        inputs=[speaker2, num_phrases2, mode2],
        outputs=[corr_heatmap, lines_plot, corr_status]
    )
    # Conditional visibility
    mode2.change(
        update_visibility,
        inputs=[mode2],
        outputs=[speaker2, num_phrases2, file_uploader, upload_btn, process_custom_btn, custom_list, delete_btn]
    )

if __name__ == "__main__":
    demo.launch()
