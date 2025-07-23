## Requirements
* Python 3.9 or greater

Unlike openai-whisper, FFmpeg does **not** need to be installed on the system. The audio is decoded with the Python library [PyAV](https://github.com/PyAV-Org/PyAV) which bundles the FFmpeg libraries in its package.

### GPU

GPU execution requires the following NVIDIA libraries to be installed:

* [cuBLAS for CUDA 12](https://developer.nvidia.com/cublas)
* [cuDNN 9 for CUDA 12](https://developer.nvidia.com/cudnn)

**Note**: The latest versions of `ctranslate2` only support CUDA 12 and cuDNN 9. For CUDA 11 and cuDNN 8, the current workaround is downgrading to the `3.24.0` version of `ctranslate2`, for CUDA 12 and cuDNN 8, downgrade to the `4.4.0` version of `ctranslate2`, (This can be done with `pip install --force-reinstall ctranslate2==4.4.0` or specifying the version in a `requirements.txt`).

There are multiple ways to install the NVIDIA libraries mentioned above. The recommended way is described in the official NVIDIA documentation, but we also suggest other installation methods below. 

## Installation

The module can be installed from [PyPI](https://pypi.org/project/faster-whisper/):

```bash
pip install faster-whisper
```

### Install Pytorch

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
Download other versions of Pytorch: https://pytorch.org/get-started/locally/
## Usage

### faster-whisper_v1.py

Upload MP3 file, then transcribe or translate.

```python
from faster_whisper import WhisperModel
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model_size = "medium"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

# task: transcribe(auto detect language) or translate to English
segments, info = model.transcribe("CantoneseSong.mp3", beam_size=5, task="transcribe")

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

### faster-whisper_v2.py

The module can transcribe a sound input in real-time within 5 seconds.

```python
import pip

#pip.main(['install', 'pyaudio'])
import pyaudio

#pip.main(['install', 'wave'])
import wave

#pip.main(['install', 'tempfile'])
import tempfile

#pip.main(['install', 'os'])
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#pip.main(['install', 'faster-whisper'])
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
```
### faster-whisper_v3.py

The module can transcribe a sound input in real-time when the person stops speaking.

```python
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
```
