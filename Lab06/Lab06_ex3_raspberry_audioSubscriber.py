from DoSomething import DoSomething
import time
import json
import pyaudio
from io import BytesIO
import numpy as np
from scipy import signal
import wave
import base64
from datetime import datetime

class Subscriber(DoSomething):
	def notify(self, topic, msg):
		input_json = json.loads(msg)
		print(topic, "Record:", input_json["recording"])

		chunk = 2400
		resolution = pyaudio.paInt16
		samp_rate = 48000
		record_secs = 1
		dev_index = 0
		chunks = int((samp_rate/chunk)*record_secs)

		frames = []
		audio = pyaudio.PyAudio()
		
		now = datetime.now()
		timestamp = int(now.timestamp())

		print("Start recording")

		stream = audio.open(format=resolution, rate= samp_rate, channels=1,
							input_device_index=dev_index, input=True,
							frames_per_buffer=chunk)

		for i in range(chunks): 
			data = stream.read(chunk)
			frames.append(data)    
		stream.stop_stream()
		stream.close()

		print("Stop recording")

		audio = np.frombuffer(b''.join(frames), dtype=np.int16)
		audio = signal.resample_poly(audio, 1, 48000/16000)
		audio = audio.astype(np.int16)
		buf = BytesIO()

		wavefile = wave.open(buf, 'wb')
		wavefile.setnchannels(1)
		wavefile.setsampwidth(2)
		wavefile.setframerate(16000)
		wavefile.writeframes(audio.tobytes())
		wavefile.close()
		buf.seek(0)

		audio_b64bytes = base64.b64encode(buf.read())
		audio_string = audio_b64bytes.decode()

		body = {
			# my url
			"bn": "http://192.168.1.92/",
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

		body = json.dumps(body)

		pub = DoSomething("audioPublisher")
		pub.run()

		# non dovrebbe essere pub?
		#test.myMqttClient.myPublish("/276033/audio", body)
		pub.myMqttClient.myPublish("/276033/audio", body)

		pub.end()



if __name__ == "__main__":
	test = Subscriber("recordAudioSubscriber")
	test.run()
	test.myMqttClient.mySubscribe("/276033/audio_recording")

	while True:
		time.sleep(1)

	test.end()