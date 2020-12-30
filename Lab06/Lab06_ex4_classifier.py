from DoSomething import DoSomething
import time
import json
from datetime import datetime


class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        events = input_json["e"]
        print(topic, events)


if __name__ == "__main__":
	test = Subscriber("thClassifier")
	test.run()
	test.myMqttClient.mySubscribe("/276033/th_classifier")

	while True:
		time.sleep(1)

	test.end()