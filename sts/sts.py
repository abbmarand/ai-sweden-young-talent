import torch
from transformers import pipeline
import pyaudio
import numpy as np
import threading
import time
from IPython.display import clear_output, display

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-tiny", device=device,
    generate_kwargs={"language": "english"}
)

sample_rate = 16000
audio_buffer = np.zeros(0, dtype=np.float32)
buffer_lock = threading.Lock()

def create_audio_stream(sample_rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024,
                    stream_callback=audio_callback)
    return p, stream

def audio_callback(in_data, frame_count, time_info, status):
    global audio_buffer
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    with buffer_lock:
        audio_buffer = np.concatenate((audio_buffer, audio_data))
        # Keep only the last 5 seconds of audio
        if len(audio_buffer) > sample_rate * 5:
            audio_buffer = audio_buffer[-sample_rate*5:]
    return (in_data, pyaudio.paContinue)

@torch.inference_mode()
def process_audio(audio_data):
    result = pipe(
        {"array": audio_data, "sampling_rate": sample_rate},
        max_new_tokens=256,
        batch_size=8
    )
    recognized_text = result['text']
    clear_output(wait=True)
    print(f"Recognized: {recognized_text}")

def process_audio_worker():
    while True:
        time.sleep(0.5)  # Adjust the sleep time as needed
        with buffer_lock:
            if len(audio_buffer) >= sample_rate * 5:
                # Always take the last 5 seconds of audio
                audio_data = audio_buffer.copy()
            else:
                continue  # Not enough data yet
        process_audio(audio_data)

def realtime_speech_recognition():
    p, stream = create_audio_stream(sample_rate=sample_rate)
    
    processing_thread = threading.Thread(target=process_audio_worker)
    processing_thread.daemon = True
    processing_thread.start()
    
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# Start real-time speech recognition
realtime_speech_recognition()
