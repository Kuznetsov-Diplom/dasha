import gradio as gr
import numpy as np
import random
from pathlib import Path
from dotenv import load_dotenv
import plotly.graph_objects as go

load_dotenv()

# === ИМПОРТЫ ===
from cv_ru_loader import load_speakers_and_phrases
from pipeline import process_phrase

speakers = load_speakers_and_phrases()
print(f"✅ Загружено {len(speakers)} спикеров из датасета")

# Хранилище пользовательских файлов: [(original_name, full_path), ...]
custom_files = []      # список кортежей
custom_vectors = []    # список обработанных векторов

def get_random_speaker():
    return random.choice(list(speakers.keys())) if speakers else None

def add_custom_files(files):
    global custom_files
    if not files:
        return [[name] for name, _ in custom_files]
    
    for file in files:
        if file is None:
            continue
        original_name = Path(file.name).name if hasattr(file, 'name') else Path(str(file)).name
        full_path = str(file.name) if hasattr(file, 'name') else str(file)
        custom_files.append((original_name, full_path))
    
    return [[name] for name, _ in custom_files]

def delete_custom_file(selected_row):
    global custom_files, custom_vectors
    if selected_row and len(selected_row) > 0:
        filename = selected_row[0][0]
        for i, (name, path) in enumerate(custom_files):
            if name == filename:
                del custom_files[i]
                if i < len(custom_vectors):
                    del custom_vectors[i]
                break
    return [[name] for name, _ in custom_files]

def process_custom_files():
    global custom_vectors
    custom_vectors = []
    success = 0
    for name, path in custom_files:
        try:
            result = process_phrase(path)
            vec = np.array(result["normalized_vector"])
            custom_vectors.append(vec)
            success += 1
        except Exception as e:
            print(f"Ошибка обработки {name}: {e}")
    return f"✅ Успешно обработано {success} из {len(custom_files)} файлов"

def correlation_own(speaker_id, num_phrases, mode):
    global custom_vectors
    vectors = []

    # Датасет
    if mode in ["Датасет", "Датасет + мои файлы"]:
        if speaker_id in speakers:
            speaker_data = speakers[speaker_id]['phrases']
            selected = speaker_data[:int(num_phrases)]
            for item in selected:
                try:
                    result = process_phrase(item['audio_path'])
                    vectors.append(np.array(result["normalized_vector"]))
                except:
                    continue

    # Пользовательские файлы
    if mode in ["Мои файлы", "Датасет + мои файлы"]:
        vectors.extend(custom_vectors)

    if len(vectors) < 2:
        return None, None, "❌ Недостаточно данных (минимум 2 вектора)"

    vectors = np.array(vectors)
    mean_vec = np.mean(vectors, axis=0)
    corr_matrix = np.corrcoef(vectors)

    # Корреляция с эталоном
    corr_with_etalon = np.array([np.corrcoef(vectors[i], mean_vec)[0, 1] for i in range(len(vectors))])
    extended_matrix = np.hstack([corr_matrix, corr_with_etalon.reshape(-1, 1)])

    # Метрика
    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
    avg_corr = float(np.mean(corr_matrix[mask])) if np.any(mask) else 0.0
    stability = "Отличная" if avg_corr > 0.85 else "Хорошая" if avg_corr > 0.75 else "Средняя" if avg_corr > 0.65 else "Низкая"

    # Heatmap
    labels = [f"Запись {i+1}" for i in range(len(vectors))] + ["Эталон"]
    fig_heat = go.Figure(data=go.Heatmap(
        z=extended_matrix,
        x=labels,
        y=labels[:-1],
        colorscale='RdYlBu_r',
        text=np.round(extended_matrix, 3),
        texttemplate="%{text}",
    ))
    fig_heat.update_layout(
        title=f'Корреляция + сравнение с эталоном ({len(vectors)} записей)',
        height=650,
        xaxis_title="Запись / Эталон",
        yaxis_title="Запись"
    )

    # Line plot
    fig_lines = go.Figure()
    for i, vec in enumerate(vectors):
        fig_lines.add_trace(go.Scatter(x=list(range(26)), y=vec, mode='lines', name=f'Запись {i+1}', opacity=0.75))
    fig_lines.add_trace(go.Scatter(x=list(range(26)), y=mean_vec, mode='lines', name='Средний вектор (эталон)', line=dict(color='black', width=4)))
    fig_lines.update_layout(
        title="Нормализованные векторы + средний вектор",
        xaxis_title="Индекс признака (0-25)",
        yaxis_title="Значение [0, 1]",
        height=500,
        legend_title="Записи"
    )

    status = f"""
    ✅ **Анализ завершён** ({len(vectors)} записей)  
    **Средняя корреляция:** {avg_corr:.3f}  
    **Стабильность:** {stability}
    """
    return fig_heat, fig_lines, status

# ==================== ИНТЕРФЕЙС ====================
with gr.Blocks(title="Dasha — Система биометрической обработки речи", theme=gr.themes.Base()) as demo:
    gr.Markdown("# Dasha — Система биометрической обработки речи")

    with gr.Tabs():
        with gr.Tab("2. Обнаружение корреляции среди своих"):
            with gr.Row():
                random_btn = gr.Button("🎲 Случайный спикер из датасета", variant="primary")
                mode_radio = gr.Radio(["Датасет", "Мои файлы", "Датасет + мои файлы"], value="Мои файлы", label="Что анализировать")

            speaker_dropdown = gr.Dropdown(choices=list(speakers.keys()), label="Спикер из датасета")
            num_phrases_slider = gr.Slider(2, 50, 10, step=1, label="Количество фраз из датасета")

            with gr.Group():
                gr.Markdown("### Загрузить свои аудиофайлы")
                file_uploader = gr.File(label="Выберите аудиофайлы (.wav, .mp3, .adts и др.)", file_count="multiple", type="filepath")
                upload_btn = gr.Button("➕ Добавить файлы", variant="secondary")
                process_custom_btn = gr.Button("🔬 Обработать загруженные файлы", variant="primary")

            custom_list = gr.Dataframe(headers=["Файл"], value=[], label="Загруженные файлы")
            delete_btn = gr.Button("🗑 Удалить выбранный файл")

            corr_btn = gr.Button("📊 Вычислить корреляцию", variant="primary", size="large")
            corr_status = gr.Markdown()
            corr_heatmap = gr.Plot()
            lines_plot = gr.Plot()

    # События
    random_btn.click(get_random_speaker, outputs=speaker_dropdown)
    upload_btn.click(add_custom_files, inputs=file_uploader, outputs=custom_list)
    process_custom_btn.click(process_custom_files, outputs=corr_status)
    delete_btn.click(delete_custom_file, inputs=custom_list, outputs=custom_list)

    corr_btn.click(
        correlation_own,
        inputs=[speaker_dropdown, num_phrases_slider, mode_radio],
        outputs=[corr_heatmap, lines_plot, corr_status]
    )

if __name__ == "__main__":
    demo.launch()