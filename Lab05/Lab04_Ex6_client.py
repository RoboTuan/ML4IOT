import base64
import datetime
import requests
import pyaudio
from io import BytesIO
import numpy as np
from scipy import signal
import wave
import argparse
import time
import json


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="dscnn")

args = parser.parse_args()



chunk = 2400
resolution = pyaudio.paInt16
samp_rate = 48000
record_secs = 1
dev_index = 0
chunks = int((samp_rate/chunk)*record_secs)

frames = []
audio = pyaudio.PyAudio()

now = datetime.datetime.now()
timestamp = int(now.timestamp())

stream = audio.open(format=resolution, rate= samp_rate, channels=1,
                    input_device_index=dev_index, input=True,
                    frames_per_buffer=chunk)

for i in range(chunks): 
    data = stream.read(chunk)
    frames.append(data)    
stream.stop_stream()


audio = np.frombuffer(b''.join(frames), dtype=np.int16)
audio = signal.resample_poly(audio, 1, 48000/16000)
audio = audio.astype(np.int16)
buf = BytesIO()

wavefile = wave.open(buf, 'wb')
wavefile.setnchannels(1)
wavefile.setsamplewidth(2)
wavefile.setframerate(16000)
wavefile.writeframes(audio.tobytes())
wavefile.close()
buf.seek(0)

audio_b64bytes = base64.b64encode(buf.read())
audio_string = audio_b64bytes.decode()

body = {
    # my url
    "bn": "http://192.168.1.232/",
    "bt": timestamp,
    "e": [
        {
            "n": "audio",
            "u": "/",
            "t": 0,
            "vd": audio_string
        }
    ]
}

url = "http://0.0.0.0:8080/{}".format(args.model)

# I don't need to manually convert body in a json if I use the
# json parameter in the put request
r = requests.put(url, json=body)

if r.status_code == 200:
    rbody = r.json()
    prob = rbody['probability']
    label = rbody['label']
    print("{} ({}%)".foramt(label, prob))
else:
    print("Error")
    print(r.text)



print(audio_string)



