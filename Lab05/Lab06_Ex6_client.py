import base64
import datetime
import requests
import pyaudio
import io
import numpy as np
from scipy import signal

import time
import json


url = "http://0.0.0.0:8080/"


audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, rate= 48000, channels=1,
                    input_device_index=0, input=True,
                    frames_per_buffer=4800)

stream.stop_stream()


frames_io = io.BytesIO()

stream.start_stream()

for i in range(0, int(48000 / 4800 * 1)): 
    data = stream.read(4800)
    frames_io.write(data)    

stream.stop_stream()

frames_io_buf = frames_io.getvalue()
frame = np.frombuffer(frames_io_buf, dtype=np.uint16)
frames_io.close()

audio2 = signal.resample_poly(frame, 16000, 48000)

audio_b64bytes = base64.b64encode(b''.join(audio))
audio_string = audio_b64bytes.decode()

print(audio_string)



