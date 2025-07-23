# environment settings
import pyaudio                          # pip install pyaudio
import numpy as np                      # pip install numpy
import wave                             # pip install wave
import os                               # pip install os
import tempfile                         # pip install tempfile
from faster_whisper import WhisperModel # pip install faster-whisper
import webrtcvad                        # pip install webrtcvad
                                        # pip install cuda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration for PyAudio
FORMAT = pyaudio.paInt16      # 16-bit int sampling format
CHANNELS = 1                  # Mono audio
RATE = 16000                 # Sampling rate expected by Whisper (16kHz)
CHUNK = 1024                 # Buffer size

# Initialize PyAudio and Whisper
p = pyaudio.PyAudio()
model = WhisperModel("medium", device="cuda", compute_type="float16")
vad = webrtcvad.Vad(2) # Set VAD mode (0-3, 3 is most aggressive)

# Open microphone stream
def is_speech(frame, sample_rate):
    return vad.is_speech(frame, sample_rate)

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Starting streaming microphone transcription. Press Ctrl+C to stop.")

try:
    while True:
        frames = []
        silence_chunks = 0
        speaking = False

        print("\nListening for speech...Press Ctrl+C to stop.")

        while True:
            data = stream.read(CHUNK)
            # VAD expects 20ms, 30ms, or 10ms frames. 1024 samples at 16kHz is 64ms, so trim or split as needed.
            frame = data[:640]  # 20ms at 16kHz, 16-bit mono = 640 bytes
            if is_speech(frame, RATE):
                frames.append(data)
                silence_chunks = 0
                speaking = True
            else:
                if speaking:
                    silence_chunks += 1
                    frames.append(data)
                # Stop after 1 second of silence (adjust as needed)
                if silence_chunks > int(RATE / CHUNK * 1):
                    break

        if frames:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                wf = wave.open(tmp_file.name, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

            # Transcribe audio chunk with faster-whisper
            segments, info = model.transcribe(tmp_file.name, beam_size=5)

            # Print detected language
            print(f"🌐 Language detected: {info.language} (Confidence: {info.language_probability:.2f})")

            # Print transcription segments
            for segment in segments:
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        # Remove temporary audio file
        os.remove(tmp_file.name)

except KeyboardInterrupt:
    print("\n✅ Stopped transcription.")

finally:
    # Clean up PyAudio stream
    stream.stop_stream()
    stream.close()
    p.terminate()
