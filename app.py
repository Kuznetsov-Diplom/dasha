import gradio as gr
import torch
import torchaudio
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# TODO: Реализация полного pipeline по Главе 2 диплома (Pre-emphasis, VAD, Normalization, Framing, MFCC, aggregation to 26-dim)
class AudioPipeline:
    def process(self, waveform, sample_rate=16000):
        # Dummy for now - later full Chapter 2 implementation
        # waveform: torch.Tensor (1D or 2D)
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        features = np.random.rand(26).tolist()
        debug_info = {
            'steps': 'Плейсхолдер. Скоро: Pre-emphasis + VAD + Нормализация + MFCC (13) + mean/std (26-dim)',
            'shape': str(waveform.shape),
            'sr': sample_rate
        }
        return features, debug_info

pipeline = AudioPipeline()

dataset = None
speakers = {}

def load_demo_data():
    global dataset, speakers
    try:
        print("Загрузка русского датасета Common Voice (ru)...")
        # Russian Common Voice - phrases from different speakers
        dataset = load_dataset("mozilla-foundation/common_voice_17_0", "ru", split="train", streaming=True)
        speakers = {}
        count = 0
        for example in list(dataset.take(300)):  # Enough to get several speakers with multiple phrases
            client_id = example.get('client_id', f'demo_{count}')
            if client_id not in speakers:
                speakers[client_id] = []
            audio = example['audio']
            if len(speakers[client_id]) < 5 and len(audio['array']) > 16000:  # at least 1 sec
                speakers[client_id].append(audio)
            count += 1
            if len(speakers) > 20:  # Limit number of speakers for demo performance
                break
        print(f"Загружено {len(speakers)} русскоязычных спикеров")
    except Exception as e:
        print(f"Ошибка загрузки датасета: {e}")
        # Fallback
        speakers = {'demo_ru_1': [{
            'array': np.random.randn(16000 * 4).astype(np.float32),
            'sampling_rate': 16000
        }] * 3}

load_demo_data()

def resample_audio(audio_dict, target_sr=16000):
    """Resample audio to 16 kHz"""
    array = audio_dict['array']
    sr = audio_dict['sampling_rate']
    if sr == target_sr:
        return array
    waveform = torch.from_numpy(array).float().unsqueeze(0)
    resampled = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    return resampled.squeeze().numpy()

def process_audio(speaker_id, phrase_idx=0, custom_audio=None):
    """Process selected audio from dataset or custom upload. Returns playable audio."""
    if custom_audio is not None:
        # custom_audio from Gradio is tuple (sr, array) or array
        if isinstance(custom_audio, tuple):
            sr, array = custom_audio
            waveform = np.array(array).astype(np.float32)
            current_sr = sr
        else:
            waveform = np.array(custom_audio).astype(np.float32)
            current_sr = 16000  # assume
    else:
        audios = speakers.get(speaker_id, [])
        if not audios:
            audios = list(speakers.values())[0] if speakers else []
        audio_dict = audios[phrase_idx % len(audios)] if audios else speakers[list(speakers.keys())[0]][0]
        waveform = resample_audio(audio_dict)
        current_sr = 16000
    
    # Convert to torch
    waveform_tensor = torch.from_numpy(waveform).float()
    if waveform_tensor.dim() == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)
    
    # Limit length for demo
    max_samples = 16000 * 30
    if waveform_tensor.shape[1] > max_samples:
        waveform_tensor = waveform_tensor[:, :max_samples]
    
    # Process with pipeline
    features, debug = pipeline.process(waveform_tensor.squeeze(0), sample_rate=16000)
    
    # Generate plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(waveform[:len(waveform)//2])  # first part
    ax[0].set_title('Waveform (Russian speech)')
    ax[0].set_xlabel('Samples')
    ax[1].specgram(waveform, Fs=16000)
    ax[1].set_title('Spectrogram')
    ax[1].set_xlabel('Time')
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    plot_img = buf.read()
    
    # Playable audio
    audio_to_play = (16000, waveform)
    
    return (
        audio_to_play,  # for listening
        plot_img,
        features,
        f"Speaker: {speaker_id}\nPhrase idx: {phrase_idx}\n{debug}"
    )

with gr.Blocks(title="Dasha Demo - Глава 2 (Русский)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""# 🎙️ Dasha — Демоверсия (Глава 2)
**Предварительная обработка + извлечение признаков**

Теперь с **русскоязычными** голосами из Common Voice. Выберите спикера — послушайте его фразы.""" )
    
    with gr.Row():
        with gr.Column(scale=2):
            speaker_id = gr.Dropdown(
                choices=list(speakers.keys())[:15], 
                label="Спикер (русский Common Voice)", 
                value=list(speakers.keys())[0] if speakers else None
            )
            phrase_idx = gr.Slider(0, 4, value=0, step=1, label="Индекс фразы (0-4)")
            btn = gr.Button("🎧 Прослушать + Обработать", variant="primary")
        with gr.Column():
            custom_upload = gr.Audio(label="Загрузить свой русский аудиофайл", type="numpy", sources=["upload", "microphone"])
    
    with gr.Row():
        audio_out = gr.Audio(label="🔊 Прослушивание выбранного трека", type="numpy")
        plot_out = gr.Image(label="Waveform + Spectrogram")
    
    features_out = gr.JSON(label="26-мерный вектор признаков (mean + std MFCC)")
    debug_out = gr.Textbox(label="Инфо / Debug")
    
    btn.click(
        fn=process_audio,
        inputs=[speaker_id, phrase_idx, custom_upload],
        outputs=[audio_out, plot_out, features_out, debug_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
