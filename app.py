import gradio as gr
import torch
import torchaudio
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# TODO: Полная реализация по Главе 2
class AudioPipeline:
    def process(self, waveform, sample_rate=16000):
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        # Dummy 26-dim пока
        features = np.concatenate([np.random.randn(13), np.random.randn(13)]).tolist()
        debug = {'steps': 'Pre-emphasis + VAD + Hamming + 13 MFCC + mean/std → 26-dim (скоро реал)', 'shape': str(waveform.shape)}
        return features, debug

pipeline = AudioPipeline()

speakers = {}

def load_russian_data():
    global speakers
    try:
        ds = load_dataset("mozilla-foundation/common_voice_17_0", "ru", streaming=True, split="train")
        speakers = {}
        count = 0
        for ex in ds:
            if count > 200: break
            cid = ex.get('client_id', f'demo_{count}')
            if cid not in speakers:
                speakers[cid] = []
            if len(speakers[cid]) < 6 and len(ex['audio']['array']) > 8000:
                speakers[cid].append(ex['audio'])
            count += 1
            if len(speakers) >= 8: break
        print(f'Загружено {len(speakers)} русских спикеров')
    except Exception as e:
        print('Fallback demo data')
        speakers = {'demo_ru': [{'array': np.random.randn(16000*5).astype(np.float32), 'sampling_rate': 16000}] * 5}

load_russian_data()

def process_audio(speaker_id, phrase_idx=0, custom_audio=None):
    try:
        if custom_audio is not None:
            sr, array = custom_audio if isinstance(custom_audio, tuple) else (16000, custom_audio)
            waveform = np.array(array, dtype=np.float32)
            text = 'Загруженный файл'
        else:
            audios = speakers.get(speaker_id, list(speakers.values())[0])
            audio_dict = audios[phrase_idx % len(audios)]
            waveform = audio_dict['array'].astype(np.float32)
            text = 'Русская фраза Common Voice'

        # Resample 16kHz
        if 'sampling_rate' in audio_dict and audio_dict['sampling_rate'] != 16000:
            res = torchaudio.transforms.Resample(audio_dict['sampling_rate'], 16000)(torch.from_numpy(waveform).unsqueeze(0))
            waveform = res.squeeze().numpy()

        features, debug = pipeline.process(waveform)

        # Plots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(waveform)
        axs[0].set_title('Waveform')
        axs[1].specgram(waveform, Fs=16000)
        axs[1].set_title('Spectrogram')
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        plot_img = Image.open(buf)

        return text, (16000, waveform), plot_img, features, str(debug)
    except Exception as e:
        return f'Error: {str(e)}', None, None, None, str(e)

with gr.Blocks(title="Dasha — Глава 2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Dasha — Русские голоса (Глава 2 demo)")
    with gr.Row():
        speaker_dd = gr.Dropdown(choices=list(speakers.keys()), label="Русский спикер", value=list(speakers.keys())[0] if speakers else None)
        phrase_slider = gr.Slider(0, 5, 0, step=1, label="Фраза")
    upload = gr.Audio(type="numpy", label="Или свой файл")
    btn = gr.Button("Обработать + Прослушать", variant="primary")
    with gr.Row():
        text_out = gr.Textbox(label="Текст")
        audio_out = gr.Audio(label="Аудио")
        plot_out = gr.Image(label="Визуализация")
        feat_out = gr.JSON(label="26-мерный вектор")
        debug_out = gr.Textbox(label="Debug")
    btn.click(process_audio, [speaker_dd, phrase_slider, upload], [text_out, audio_out, plot_out, feat_out, debug_out])

demo.launch(server_name="0.0.0.0", server_port=7860)
