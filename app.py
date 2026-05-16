import gradio as gr
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from pipeline import process_phrase
from cv_ru_loader import load_speakers_and_phrases

speakers = load_speakers_and_phrases()

# ====================== STATE ======================
custom_files_state = gr.State([])      # [(name, path), ...]
custom_vectors_state = gr.State([])    # [np.array, ...]

def add_custom_files(files, current_files, current_vectors):
    new_files = current_files.copy()
    new_vectors = current_vectors.copy()

    for f in files or []:
        name = Path(f.name).name
        path = str(f.name)
        new_files.append((name, path))
        
        try:
            result = process_phrase(path)
            vec = np.array(result['normalized_vector'])
            new_vectors.append(vec)
            print(f'✅ Обработан {name} | norm = {np.linalg.norm(vec):.4f}')
        except Exception as e:
            print(f'❌ Ошибка {name}: {e}')

    return [[n for n, _ in new_files]], new_files, new_vectors

def delete_custom_file(selected_idx, current_files, current_vectors):
    if not selected_idx or not current_files:
        return current_files, current_vectors
    idx = selected_idx[0][0] if isinstance(selected_idx[0], list) else selected_idx[0]
    del current_files[idx]
    if idx < len(current_vectors):
        del current_vectors[idx]
    return [[n for n, _ in current_files]], current_files, current_vectors

def update_visibility(mode):
    dataset_visible = mode == "Датасет"
    custom_visible = mode == "Мои файлы"
    return (
        gr.update(visible=dataset_visible),
        gr.update(visible=dataset_visible),
        gr.update(visible=custom_visible),
        gr.update(visible=custom_visible),
        gr.update(visible=custom_visible),
        gr.update(visible=custom_visible),
    )

def correlation_own(speaker_id, num_phrases, mode, custom_files, custom_vectors):
    vectors = []
    labels = []

    if mode == "Датасет":
        if speaker_id in speakers:
            data = speakers[speaker_id]['phrases'][:int(num_phrases)]
            for item in data:
                try:
                    result = process_phrase(item['audio_path'])
                    vectors.append(np.array(result['normalized_vector']))
                    labels.append(item['sentence'][:30] + "...")
                except:
                    continue
    else:
        vectors = [np.array(v) for v in custom_vectors]
        labels = [name for name, _ in custom_files]

    if len(vectors) < 2:
        return None, None, "Нужно минимум 2 записи"

    vectors = np.array(vectors)
    mean_vec = np.mean(vectors, axis=0)

    corr_matrix = np.corrcoef(vectors)
    corr_with_mean = np.array([np.corrcoef(v, mean_vec)[0,1] for v in vectors])
    extended = np.hstack([corr_matrix, corr_with_mean.reshape(-1, 1)])

    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
    avg_corr = float(np.mean(corr_matrix[mask])) if np.any(mask) else 1.0
    stability = "Отличная" if avg_corr > 0.85 else "Хорошая" if avg_corr > 0.75 else "Средняя" if avg_corr > 0.65 else "Низкая"

    labels_with_etalon = labels + ["Эталон"]

    fig_heat = go.Figure(data=go.Heatmap(
        z=extended,
        x=labels_with_etalon,
        y=labels,
        colorscale='RdYlBu_r',
        text=np.round(extended, 3),
        texttemplate="%{text:.2f}",
    ))
    fig_heat.update_layout(title=f'Корреляция + эталон ({len(vectors)} записей)', height=650)

    fig_lines = go.Figure()
    for i, v in enumerate(vectors):
        fig_lines.add_trace(go.Scatter(x=list(range(26)), y=v, mode='lines', name=labels[i]))
    fig_lines.add_trace(go.Scatter(x=list(range(26)), y=mean_vec, mode='lines', name='Средний вектор (эталон)', line=dict(color='black', width=4)))
    fig_lines.update_layout(title="Векторы + средний вектор", height=400)

    status = f"**Средняя корреляция:** {avg_corr:.3f}<br>**Стабильность:** {stability}"
    return fig_heat, fig_lines, status

with gr.Blocks(title="Dasha — Биометрия", theme=gr.themes.Dark()) as demo:
    gr.Markdown("# Dasha — Система биометрической обработки речи")

    with gr.Tabs():
        with gr.Tab("2. Обнаружение корреляции среди своих"):
            with gr.Row():
                with gr.Column(scale=1):
                    mode2 = gr.Radio(["Датасет", "Мои файлы"], value="Мои файлы", label="Что анализировать")

                    speaker2 = gr.Dropdown(choices=list(speakers.keys()), label="Спикер", visible=False)
                    num_phrases2 = gr.Slider(2, 50, value=10, step=1, label="Количество фраз", visible=False)

                    gr.Markdown("### Загрузить свои файлы")
                    file_uploader = gr.File(file_count="multiple", label="Выберите аудиофайлы")
                    upload_btn = gr.Button("➕ Добавить файлы", variant="secondary")
                    
                    custom_list = gr.Dataframe(headers=["Файл"], value=[], label="Загруженные файлы", interactive=True)
                    delete_btn = gr.Button("🗑 Удалить выбранный", variant="stop")

                    gr.Markdown("---")
                    corr_btn = gr.Button("📊 Вычислить корреляцию", variant="primary", size="large")

                with gr.Column(scale=2):
                    corr_status = gr.Markdown("### Результат появится здесь")
                    corr_heatmap = gr.Plot()
                    lines_plot = gr.Plot()

    mode2.change(update_visibility, inputs=mode2, outputs=[speaker2, num_phrases2, file_uploader, upload_btn, custom_list, delete_btn])

    upload_btn.click(add_custom_files, inputs=[file_uploader, custom_files_state, custom_vectors_state], outputs=[custom_list, custom_files_state, custom_vectors_state])

    delete_btn.click(delete_custom_file, inputs=[custom_list, custom_files_state, custom_vectors_state], outputs=[custom_list, custom_files_state, custom_vectors_state])

    corr_btn.click(correlation_own, inputs=[speaker2, num_phrases2, mode2, custom_files_state, custom_vectors_state], outputs=[corr_heatmap, lines_plot, corr_status])

if __name__ == "__main__":
    demo.launch()