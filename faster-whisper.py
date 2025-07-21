import pip
pip.main(['install', 'pyaudio'])
pip.main(['install', 'numpy'])
pip.main(['install', 'wave'])
pip.main(['install', 'tempfile'])
pip.main(['install', 'torchaudio'])
pip.main(['install', 'torch'])
pip.main(['install', 'os'])
pip.main(['install', 'faster-whisper'])

import pyaudio
import wave
import tempfile
import os
from faster_whisper import WhisperModel

# Configuration for PyAudio
FORMAT = pyaudio.paInt16      # 16-bit int sampling format
CHANNELS = 1                  # Mono audio
RATE = 16000                 # Sampling rate expected by Whisper (16kHz)
CHUNK = 1024                 # Buffer size
RECORD_SECONDS = 5           # Duration per audio chunk

# Initialize PyAudio and Whisper
p = pyaudio.PyAudio()
model = WhisperModel("medium", device="cuda", compute_type="float16")

# Open microphone stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Starting streaming microphone transcription. Press Ctrl+C to stop.")

try:
    while True:
        print(f"\n🎤 Recording {RECORD_SECONDS} seconds...Press Ctrl+C to stop.")
        frames = []

        # Read audio data chunk by chunk
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # Save recorded chunk to a temporary WAV file
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
