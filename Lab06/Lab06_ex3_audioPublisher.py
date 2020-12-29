from DoSomething import DoSomething
import time
import json
import base64
from datetime import datetime

class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)

        audio_string = input_json["e"][0]["vd"]
        audio_bytes = base64.b64decode(audio_string)
        wav_file = open("file.wav", "wb")
        wav_file.write(audio_bytes)

        print(topic)



if __name__ == "__main__":
    test = DoSomething("recordAudioPublisher")
    test.run()

    body = {"recording": "audio"}
    body = json.dumps(body)
    
    test.myMqttClient.myPublish("/276033/audio_recording", body)
    
    sub = Subscriber("audioSubscriber")
    sub.run()
    sub.myMqttClient.mySubscribe("/276033/audio")

    time.sleep(5)

    sub.end()

    test.end()