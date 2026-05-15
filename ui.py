import gradio as gr
from typing import List, Tuple
import os

from cv_ru_loader import load_speakers_and_phrases
from pipeline import process_phrase
from feature_normalizer import FeatureNormalizer


def shorten_speaker_id(speaker_id: str, length: int = 8) -> str:
    '''Сокращаем client_id, оставляем последние length символов'''
    if len(speaker_id) > length:
        return speaker_id[-length:]
    return speaker_id


def get_speakers_list() -> List[Tuple[str, str]]:
    '''Возвращает список (display_name, speaker_id)'''
    try:
        speakers = load_speakers_and_phrases()
        speakers_list = []
        for speaker_id, phrases in speakers.items():
            short_id = shorten_speaker_id(speaker_id)
            num_phrases = len(phrases)
            display = f"{short_id} ({num_phrases} фраз)"
            speakers_list.append((display, speaker_id))
        # Сортируем по display name
        speakers_list.sort(key=lambda x: x[0])
        return speakers_list
    except Exception as e:
        print(f"Ошибка загрузки спикеров: {e}")
        return [("Ошибка загрузки данных", "error")]


def get_phrases_for_speaker(speaker_id: str) -> List[Tuple[str, str]]:
    '''Возвращает список (phrase_text, audio_path) для спикера'''
    if not speaker_id or speaker_id == "error":
        return [("Нет данных", "")]
    try:
        speakers = load_speakers_and_phrases()
        phrases = speakers.get(speaker_id, [])
        return [(p['sentence'], p['path']) for p in phrases]
    except:
        return [("Ошибка", "")]


def process_and_visualize(audio_path: str):
    '''Основная функция обработки с возвратом всех визуализаций'''
    if not audio_path:
        return "Выберите аудио", None, None, None, None, None, "[]"
    
    try:
        result = process_phrase(audio_path)
        
        # result должен содержать: waveform, pre_emphasis, vad, mfcc, normalized_vector и т.д.
        steps_html = """
        <h3>✅ Обработка завершена успешно!</h3>
        <p>1. Загрузка аудио — OK</p>
        <p>2. Pre-emphasis — OK</p>
        <p>3. VAD — OK</p>
        <p>4. MFCC — OK</p>
        <p>5. Нормализация [0,1] — OK</p>
        <p>6. Формирование вектора — OK</p>
        <p>7. НПБК (скелет) — OK</p>
        """
        
        normalized_str = str(result.get('normalized_vector', []))
        
        return (
            steps_html,
            result.get('waveform_plot'),
            result.get('pre_emphasis_plot'),
            result.get('mfcc_plot'),
            normalized_str,
            result.get('normalized_vector', [])
        )
    except Exception as e:
        return f"<h3 style='color:red'>Ошибка обработки: {str(e)}</h3>", None, None, None, "[]", []


def create_interface():
    with gr.Blocks(title="Dasha — Биометрия речи", theme=gr.themes.Dark()) as demo:
        gr.Markdown("""
        # Dasha — Система биометрической обработки речи
        **Одно окно для демонстрации полного пайплайна**
        Выберите спикера → фразу → обработайте по алгоритму ГОСТ
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                speaker_dropdown = gr.Dropdown(
                    label="Выберите спикера",
                    choices=[("Загрузка...", "")],
                    value="",
                    interactive=True
                )
                
                phrase_dropdown = gr.Dropdown(
                    label="Выберите фразу",
                    choices=[("Выберите спикера", "")],
                    interactive=True
                )
                
                audio_player = gr.Audio(
                    label="Прослушать аудио",
                    type="filepath",
                    interactive=False
                )
                
                process_btn = gr.Button("🚀 Обработать фразу", variant="primary", size="large")
            
            with gr.Column(scale=2):
                output_html = gr.HTML(label="Процесс обработки")
                waveform_plot = gr.Plot(label="Waveform")
                pre_plot = gr.Plot(label="Pre-emphasis")
                mfcc_plot = gr.Plot(label="MFCC признаки")
                normalized_output = gr.Textbox(label="Нормализованный вектор [0, 1]", lines=8)
                
        # События
        speaker_dropdown.change(
            fn=get_phrases_for_speaker,
            inputs=speaker_dropdown,
            outputs=phrase_dropdown
        )
        
        phrase_dropdown.change(
            fn=lambda p: p[1] if p else None,
            inputs=phrase_dropdown,
            outputs=audio_player
        )
        
        process_btn.click(
            fn=process_and_visualize,
            inputs=audio_player,
            outputs=[output_html, waveform_plot, pre_plot, mfcc_plot, normalized_output]
        )
        
        # Загрузка спикеров при старте
        demo.load(
            fn=get_speakers_list,
            inputs=None,
            outputs=speaker_dropdown
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
