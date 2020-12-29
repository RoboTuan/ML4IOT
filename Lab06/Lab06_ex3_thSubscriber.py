from DoSomething import DoSomething
import time
import json
from datetime import datetime

class Subscriber(DoSomething):
	def notify(self, topic, msg):
		input_json = json.loads(msg)
		timestampt = input_json["bt"]
		date = datetime.fromtimestamp(timestampt)
		event = input_json[0]
		print(topic, date, event)

if __name__ == "__main__":
	test = Subscriber("thSubscriber")
	test.run()
	test.myMqttClient.mySubscribe("/276033/th")

	while True:
		time.sleep(1)

	test.end()