from DoSomething import DoSomething
import base64
import datetime
import pyaudio
import time
import json

class CollectorClient(DoSomething):
    def __init__(self, clientID):
        super().__init__(clientID)
        audio = pyaudio.PyAudio()
        self.stream = audio.open(format=pyaudio.paInt16, rate=48000, channels=1,
                            input_device_index=0, input=True,
                            frames_per_buffer=4800)
        self.stream.stop_stream()

    def notify(self, topic, msg):
        senml = json.loads(msg)
        events = senml['e']
        for event in events:
            if event['n']=='record':
                if event['vb'] is True:
                    now = datetime.datetime.now()
                    timestamp = int(now.timestamp())
                    self.stream.start_stream()
                    frames = []
                    print("Recording")
                    for ii in range(10):
                        data = self.stream.read(4800)
                        frames.append(data)
                    self.stream.stop_stream()
                    audio_b64bytes = base64.b64encode(b''.join(frames))
                    audio_string = audio_b64bytes.decode()
                    print("Stop recording")

                    output = {
                        "bn": "http://192.168.1.92/",
                        "bt": timestamp, 
                        "e": [
                            {"n": "audio", "u": "/", "t": 0, "vd": audio_string}
                        ]
                    }

                    output = json.dumps(output)
                    self.myMqttClient.myPublish('/276033/audio', output)

if __name__=="__main__":
    test = CollectorClient("collector")
    test.run()
    test.myMqttClient.mySubscribe("/276033/record")

    while True:
        time.sleep(1)
