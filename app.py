import gradio as gr
import torch
import torchaudio
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Simple placeholder pipeline for demo
class AudioPipeline:
    def process(self, waveform):
        # Dummy processing
        features = np.random.rand(26).tolist()
        return features, {'steps': 'Demo steps completed'}

pipeline = AudioPipeline()

dataset = None
speakers = {}

def load_demo_data():
    global dataset, speakers
    try:
        dataset = load_dataset("speechbrain/voxceleb2", "dev", split="train", streaming=True)
        for example in list(dataset.take(100)):
            spk = example['speaker_id']
            if spk not in speakers:
                speakers[spk] = []
            if len(speakers[spk]) < 3:
                speakers[spk].append(example['audio'])
    except:
        speakers = {'demo1': [{'array': np.random.rand(16000*5), 'sampling_rate': 16000}]}

load_demo_data()

def process_audio(speaker_id, num_phrases=1, custom_audio=None):
    if custom_audio is not None:
        waveform = custom_audio[1] if isinstance(custom_audio, tuple) else custom_audio
    else:
        audios = speakers.get(speaker_id, [speakers[list(speakers.keys())[0]][0]])
        audio_dict = audios[0]
        waveform = audio_dict['array'] if isinstance(audio_dict, dict) else audio_dict
    
    waveform = torch.from_numpy(np.array(waveform)).float().unsqueeze(0)
    if waveform.shape[1] > 16000*30:
        waveform = waveform[:, :16000*30]
    
    features, debug = pipeline.process(waveform.squeeze())
    
    # Dummy plots
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(waveform.squeeze().numpy())
    ax[0].set_title('Waveform')
    ax[1].specgram(waveform.squeeze().numpy(), Fs=16000)
    ax[1].set_title('Spectrogram')
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_img = buf.read()
    
    return (
        (16000, waveform.squeeze().numpy()),  # audio
        plot_img,
        features,
        str(debug)
    )

with gr.Blocks(title="Dasha Demo - Глава 2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Dasha — Тестовая версия (Глава 2)\n**Предварительная обработка речевого сигнала + 26-мерный вектор**")
    
    with gr.Row():
        with gr.Column():
            speaker_id = gr.Dropdown(choices=list(speakers.keys())[:30], label="Speaker ID (VoxCeleb2 streaming)", value=list(speakers.keys())[0] if speakers else "demo1")
            num_phrases = gr.Slider(1, 3, value=1, label="Кол-во фраз")
            btn = gr.Button("Обработать", variant="primary")
        with gr.Column():
            custom_upload = gr.Audio(label="Или загрузить свой файл", type="numpy")
    
    with gr.Row():
        audio_out = gr.Audio(label="Оригинал для прослушивания")
        plot_out = gr.Image(label="Waveform + Spectrogram")
    
    features_out = gr.JSON(label="26-мерный вектор признаков")
    debug_out = gr.Textbox(label="Debug info")
    
    btn.click(
        fn=process_audio,
        inputs=[speaker_id, num_phrases, custom_upload],
        outputs=[audio_out, plot_out, features_out, debug_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)