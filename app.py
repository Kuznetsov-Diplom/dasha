import gradio as gr
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from pipeline import process_phrase
from cv_ru_loader import load_speakers_and_phrases

speakers = load_speakers_and_phrases()

# ====================== STATE ======================
custom_files_state = gr.State([])      # list of (name, path)
custom_vectors_state = gr.State([])    # list of np.array (26-dim)

def add_custom_files(files, current_files, current_vectors):
    """Сразу обрабатывает через process_phrase и добавляет в список"""
    new_files = current_files.copy() if current_files else []
    new_vectors = current_vectors.copy() if current_vectors else []

    processed = 0
    for f in files or []:
        try:
            name = Path(getattr(f, 'name', str(f))).name
            path = str(getattr(f, 'name', f))
            
            result = process_phrase(path)
            vec = np.array(result['normalized_vector'], dtype=np.float32).copy()  # ← независимая копия!
            
            new_files.append((name, path))
            new_vectors.append(vec)
            processed += 1
            print(f'✅ Обработан {name} | norm = {np.linalg.norm(vec):.4f}')
        except Exception as e:
            print(f'❌ Ошибка {name}: {e}')
            continue

    return [[n for n, _ in new_files]], new_files, new_vectors, f'✅ Добавлено {processed} файлов. Всего: {len(new_files)}'

def delete_custom_file(selected_idx, current_files, current_vectors):
    if not selected_idx or not current_files:
        return current_files, current_vectors, []
    idx = selected_idx[0][0] if isinstance(selected_idx[0], list) else selected_idx[0]
    if 0 <= idx < len(current_files):
        del current_files[idx]
        if idx < len(current_vectors):
            del current_vectors[idx]
    return [[n for n, _ in current_files]], current_files, current_vectors, 'Данные удалены'

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
                    vectors.append(np.array(result['normalized_vector'], dtype=np.float32).copy())
                    labels.append(item['sentence'][:30] + "...")
                except:
                    continue
    else:
        # Мои файлы - используем сохранённые векторы
        vectors = [np.array(v, dtype=np.float32).copy() for v in custom_vectors]
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

def auto_correlation(mode, custom_files, custom_vectors):
    """Автоматический расчёт при изменении списка файлов (без отдельной кнопки)"""
    if mode != "Мои файлы" or len(custom_vectors) < 2:
        return None, None, "Добавьте минимум 2 файла в режиме \"Мои файлы\""
    return correlation_own(None, 0, mode, custom_files, custom_vectors)

with gr.Blocks(title="Dasha — Биометрия", theme=gr.themes.Base()) as demo:
    gr.Markdown("# Dasha — Система биометрической обработки речи")

    with gr.Tabs():
        with gr.Tab("2. Обнаружение корреляции среди своих"):
            with gr.Row():
                with gr.Column(scale=1):
                    mode2 = gr.Radio(["Датасет", "Мои файлы"], value="Мои файлы", label="Режим")

                    speaker2 = gr.Dropdown(choices=list(speakers.keys()), label="Спикер", visible=False)
                    num_phrases2 = gr.Slider(2, 50, value=10, step=1, label="Количество фраз", visible=False)

                    gr.Markdown("### Загрузить свои файлы")
                    file_uploader = gr.File(file_count="multiple", label="Выберите аудиофайлы (.wav, .mp3)")
                    upload_btn = gr.Button("➕ Добавить файлы", variant="primary")
                    
                    custom_list = gr.Dataframe(headers=["Файл"], value=[], label="Загруженные файлы", interactive=True)
                    delete_btn = gr.Button("🗑 Удалить выбранный", variant="stop")

                    status_box = gr.Textbox(label="Статус", interactive=False)

                with gr.Column(scale=2):
                    corr_status = gr.Markdown("### Результат появится здесь")
                    corr_heatmap = gr.Plot()
                    lines_plot = gr.Plot()

    # Визибилити
    mode2.change(update_visibility, inputs=mode2, outputs=[speaker2, num_phrases2, file_uploader, upload_btn, custom_list, delete_btn])

    # Главное исправление: кнопка сразу обрабатывает и добавляет
    upload_btn.click(
        fn=add_custom_files,
        inputs=[file_uploader, custom_files_state, custom_vectors_state],
        outputs=[custom_list, custom_files_state, custom_vectors_state, status_box]
    )

    # Автоматический пересчёт корреляции при изменении списка
    custom_vectors_state.change(
        fn=auto_correlation,
        inputs=[mode2, custom_files_state, custom_vectors_state],
        outputs=[corr_heatmap, lines_plot, corr_status]
    )

    delete_btn.click(
        fn=delete_custom_file,
        inputs=[custom_list, custom_files_state, custom_vectors_state],
        outputs=[custom_list, custom_files_state, custom_vectors_state, status_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)