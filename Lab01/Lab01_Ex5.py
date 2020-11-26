import os
import pyaudio
import wave
import time as t

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "file2.wav"

audio = pyaudio.PyAudio()

print(f"Bith depth: 16")
print(f"Sample rate: {RATE}")
print(f"Chunk size: {CHUNK}")
print(f"Recording time: {RECORD_SECONDS} seconds")
print(f"Output file name: {WAVE_OUTPUT_FILENAME}")
print()

stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

print("recording...")

frames = []

start_sensing = t.time()
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
end_sensing = t.time()
sensing_time = start_sensing-end_sensing

print("finished recording")
 
 # stop Recording
stream.stop_stream()
stream.close()
# Destroy auio object to destroy memory
audio.terminate()

t_start_storing = t.time()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

t_end_storing = t.time()

print(f"time for sensing the audio: {sensing_time}")
print(f"Time for storing the data on disk: {round(t_end_storing-t_start_storing, 4)} seconds")

wav_size = os.path.getsize("./file2.wav")
print(f"The size of the wav file is: {int(wav_size/1024)} KiloBytes")
