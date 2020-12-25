import cherrypy 
import json
import base64
from cherrypy.process.wspbus import ChannelFailures
import pyaudio
from board import D4
import adafruit_dht
from picamera import PiCamera
import picamera.exc
import datetime


class Raspberry(object):
    exposed = True

    def __init__(self):
        self.dht_device = adafruit_dht.DHT11(D4)
        audio = pyaudio.PyAudio()
        self.stream = audio.open(format=pyaudio.paInt16, rate= 48000, channels=1,
                            input_device_index=0, input=True,
                            frames_per_buffer=4800)
        
        self.stream.stop_stream()
        print("Check the url in the body created in the get method below")

    def GET (self, *uri, **query):

        #return "yolo"

        now = datetime.datetime.now()
        timestamp = int(now.timestamp())

        temperature = self.dht_device.temperature
        humidity = self.dht_device.humidity

        frames = []
        self.stream.start_stream()
        for _ in range(10):
            data = self.stream.read(4800)
            frames.append(data)
        self.stream.stop_stream()
        audio_b64bytes = base64.b64encode(b''.join(frames))
        audio_string = audio_b64bytes.decode()

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
                {"n": "humidity", "u":"RH", "t":0, "v":humidity},
                #vd for array data
                {"n": "audio", "u":"/", "t":0, "vd":audio_string}
            ]
        }

        body = json.dumps(body)
        
        return body



if __name__ == '__main__':
	conf = {
		'/': {
			'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
			#'tools.sessions.on': True
		}
	}
	cherrypy.tree.mount(Raspberry(), '/', conf)

	cherrypy.config.update({'server.socket_host': '0.0.0.0'})
	cherrypy.config.update({'server.socket_port': 8080})
	cherrypy.engine.start()
	cherrypy.engine.block()