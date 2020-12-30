from DoSomething import DoSomething
import time
import json
from board import D4
import adafruit_dht
from datetime import datetime



if __name__ == "__main__":
	test = DoSomething("thSendForPrediction")
	test.run()

	dht_device = adafruit_dht.DHT11(D4)

	while True:

		now = datetime.now()
		timestamp = int(now.timestamp())

		body = {
            #identifier
            # CHECK URL 
            "bn": "http://192.168.1.92/",
            #base timestamp
            "bt": timestamp,
            #list of events
            "e":[
            ]
        }

        for i in range(6):

            temperature = dht_device.temperature
            humidity = dht_device.humidity

            #name, unit, timestamp, offset, value 
            body["e"].append({"n": "temperature", "u":"cel", "t": i*10, "v":temperature})
            body["e"].append({"n": "humidity", "u":"RH", "t": i*10, "v":humidity})
            


            

            time.sleep(10)
        
        body_json = json.dumps(body)

        test.myMqttClient.myPublish("/276033/th_classifier", body_json)


	#test.end()