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

# task: transcribe or translate
# transcribe: transcribe audio in the same language
# translate: translate audio to English
# model.transcribe("filename", beam_size=int, task="transcribe")
segments, info = model.transcribe("CantoneseSong.mp3", beam_size=5, task="transcribe")

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))