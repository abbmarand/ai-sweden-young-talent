import torch
from transformers import pipeline
import moviepy.editor as mp
import numpy as np
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import soundfile as sf
import scipy.io.wavfile
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import librosa
from TTS.api import TTS
        
def extract_audio(mp4_path, audio_path="temp_audio.wav"):
    video = mp.VideoFileClip(mp4_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    video.close()
    audio.close()
    del video, audio
    return audio_path

def mp4_to_text(audio_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v2",
        device=device,
        generate_kwargs={"language": "swedish"}
    )
    
    audio_segment = AudioSegment.from_wav(audio_path)
    chunk_length_ms = 30000  # 30 seconds
    chunks = make_chunks(audio_segment, chunk_length_ms)
    
    transcription = ""
    with torch.inference_mode():
        for i, chunk in enumerate(chunks):
            chunk_path = f"temp_chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            result = pipe(chunk_path, batch_size=8, return_timestamps=True)
            transcription += result["text"] + " "
            os.remove(chunk_path)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    
    del pipe
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    
    return transcription.strip()

def translate_text(text):


    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_chunk_length = 1024
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    translated_chunks = []
    with torch.inference_mode():
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_chunk_length).to(device)
            translated_ids = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"], max_length=max_chunk_length)
            translation = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
            translated_chunks.append(translation)

    del model, tokenizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return " ".join(translated_chunks)

def generate_speech(text, input_audio_file='./sts/temp_audio.wav', output_file="generated_speech.wav"):

    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=False,
        gpu=torch.cuda.is_available()
    )

    try:
        if input_audio_file and os.path.exists(input_audio_file):
            tts.tts_to_file(text=text, file_path=output_file, speaker_wav="/home/m/dev/ai/sts/temp_audio.wav", language="en")
        else:
            tts.tts_to_file(text=text, file_path=output_file, language="en")
       
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return None

    finally:
        del tts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    mp4_file = "/home/m/dev/ai/sts/TorsdagsDrammen 2024-04-25.mp4"
    audio_file = extract_audio(mp4_file, audio_path="temp_audio.wav")
    transcription = mp4_to_text(audio_file)
    print(f"Transcription: {transcription}")
    #translation = translate_text(transcription)
    #print(f"Translation: {translation}")
    generate_speech(transcription, input_audio_file=audio_file)