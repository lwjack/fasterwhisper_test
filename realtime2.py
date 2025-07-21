import pip
pip.main(['install', 'sounddevice'])
import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#settings 
samplerate = 16000
block_duration = 0.5 #seconds
chunk_duration = 2 #seconds
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

# Model setup: medium + float32
model = WhisperModel("medium", device="cuda", compute_type="float32")

def audio_callback(indata, frame , time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def recorder():
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback, blocksize=frames_per_block):
        print("Listening... Press Ctrl+c to stop")
        while True:
            sd.sleep(100)

def trancriber():
    global audio_buffer
    while True:
        block = audio_queue.get()
        audio_buffer.append(block)

        total_frames = sum(len(b) for b in audio_buffer)
        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            audio_buffer = [] #clear buffer

            audio_data = audio_data.flatten().astype(np.float32)

            # Transcription without timestamps
            segments, _= model.transcribe(
            audio_data,
            beam_size=1 # Max speed
            )
            for segment in segments:
                print(f"{segment.text}")
    
# Start threads
threading.Thread(target=recorder, daemon=True).start()
trancriber()