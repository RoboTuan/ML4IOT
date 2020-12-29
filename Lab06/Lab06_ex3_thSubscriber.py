from DoSomething import DoSomething
import time
import json
from datetime import datetime

class Subscriber(DoSomething):
	def notify(self, topic, msg):
		input_json = json.loads(msg)
		timestampt = input_json["bt"]
		date = datetime.fromtimestamp(timestampt)
		event = input_json["e"][0]
		measure = event["n"]
		value = event["v"]
		print(topic, date, measure, value)

if __name__ == "__main__":
	test = Subscriber("thSubscriber")
	test.run()
	test.myMqttClient.mySubscribe("/276033/temperature")
	test.myMqttClient.mySubscribe("/276033/humidity")

	while True:
		time.sleep(1)

	test.end()