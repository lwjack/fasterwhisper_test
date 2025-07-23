import pyaudio
import wave
import tempfile
import os
import numpy as np
from faster_whisper import WhisperModel

# Configuration for PyAudio
FORMAT = pyaudio.paInt16      # 16-bit int sampling format
CHANNELS = 1                  # Mono audio
RATE = 16000                  # Sampling rate expected by Whisper (16kHz)
CHUNK = 1024                  # Buffer size
SILENCE_THRESHOLD = 300       # RMS energy threshold for silence (adjust based on environment)
SILENCE_DURATION = 1.0        # Duration of silence to stop recording (in seconds)
MIN_SPEECH_DURATION = 0.5     # Minimum speech duration to process (seconds)

# Calculate silence parameters
silence_chunks = int(SILENCE_DURATION * RATE / CHUNK)
min_speech_chunks = int(MIN_SPEECH_DURATION * RATE / CHUNK)

# Initialize PyAudio and Whisper
p = pyaudio.PyAudio()
model = WhisperModel("medium", device="cuda", compute_type="float16")

# Open microphone stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Starting real-time transcription. Press Ctrl+C to stop.")
print(f"🔇 Silence threshold: {SILENCE_THRESHOLD} | Minimum speech: {MIN_SPEECH_DURATION}s")

def calculate_rms(data):
    """Calculate RMS energy of audio chunk."""
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.sqrt(np.mean(np.square(audio_data)))

try:
    frames = []
    silence_counter = 0
    recording = False
    
    while True:
        data = stream.read(CHUNK)
        rms = calculate_rms(data)
        
        # Detect speech based on energy threshold
        if rms > SILENCE_THRESHOLD:
            if not recording:
                print("\n🎤 Speech detected! Recording started...")
            recording = True
            silence_counter = 0
            frames.append(data)
        else:
            if recording:
                silence_counter += 1
                frames.append(data)  # Keep adding silence during pause
                
                # End recording after sufficient silence
                if silence_counter >= silence_chunks:
                    # Check if we have enough speech to process
                    if len(frames) > min_speech_chunks:
                        # Save recorded speech to temporary file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            wf = wave.open(tmp_file.name, 'wb')
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(p.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(b''.join(frames))
                            wf.close()

                            # Transcribe audio
                            segments, info = model.transcribe(tmp_file.name, beam_size=5)
                            
                            # Print results
                            print(f"\n✅ Transcription complete ({len(frames)*CHUNK/RATE:.1f}s):")
                            print(f"🌐 Language: {info.language} (Confidence: {info.language_probability:.2f})")
                            
                            full_text = ""
                            for segment in segments:
                                print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")
                                full_text += segment.text + " "
                            
                            print(f"📝 Full text: {full_text.strip()}")
                            
                        # Clean up
                        os.remove(tmp_file.name)
                    else:
                        print("\n⚠️  Speech too short, discarding")
                    
                    # Reset recording
                    frames = []
                    recording = False
                    silence_counter = 0
                    print("\n🔍 Listening for speech...")
            else:
                # Continuous silence, keep minimal buffer
                if len(frames) > silence_chunks:
                    frames = frames[-silence_chunks:]

except KeyboardInterrupt:
    print("\n🛑 Transcription stopped by user")

finally:
    # Clean up resources
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("✅ Resources released")