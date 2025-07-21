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

### fwcode.py

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

### realtime.py

The module can transcribe a sound input in real-time, but it needs a sound device.

```python
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
```


