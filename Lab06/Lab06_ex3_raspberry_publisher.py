from DoSomething import DoSomething
import time
import json
from board import D4
import adafruit_dht
from datetime import datetime



if __name__ == "__main__":
	test = DoSomething("temperature_humidity")
	test.run()

	dht_device = adafruit_dht.DHT11(D4)

	twenty = True 

	while True:

		now = datetime.now()
		timestamp = int(now.timestamp())

		temperature = dht_device.temperature

		body = {
            #identifier
            # CHECK URL 
            "bn": "http://192.168.1.92/",
            #base timestamp
            "bt": timestamp,
            #list of events
            "e":[
                #name, unit, timestamp, offset, value 
                {"n": "temperature", "u":"cel", "t":0, "v":temperature},
            ]
        }

		body_json = json.dumps(body)

		test.myMqttClient.myPublish("/276033/temperature", body_json)

		if twenty is False:
			twenty = True
		elif twenty is True:
			twenty = False

			humidity = dht_device.humidity

			body = {
				#identifier
				# CHECK URL 
				"bn": "http://192.168.1.92/",
				#base timestamp
				"bt": timestamp,
				#list of events
				"e":[
					#name, unit, timestamp, offset, value 
					{"n": "humidity", "u":"RH", "t":0, "v":humidity},
				]
        	}

			body_json = json.dumps(body)

			test.myMqttClient.myPublish("/276033/humidity", body_json)

		time.sleep(10)


	#test.end()