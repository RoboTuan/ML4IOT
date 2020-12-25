import cherrypy 
import json
import base64
from cherrypy.process.wspbus import ChannelFailures
import numpy as np
import tensorflow as tf



class KWSinference(object):
        exposed = True

        def __init__(self):
                self.length = 640
                self.stride = 320
                self.bins = 40
                self.ceoff = 10
                self.rate = 16000
                self.resize = 32

                self.num_frames = (self.rate - self.length) // self.stride + 1
                self.num_spectrogram_bins = self.length // 2 + 1

                self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                        self.bins, self.num_spectrogram_bins, self.rate, 20, 4000)
    

        def POST(self, *uri, audio_string):
                
                audio_bytes = base64.b64decode(audio_string)

                audio = np.frombuffer(audio_bytes, dtype=np.uint16)

                stft = tf.signal.stft(audio, self.length, self.stride, fft_length=self.stride)
                spectrogram = tf.abs(stft)

                mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
                log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
                mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
                mfccs = mfccs[..., :self.ceoff]
                mfccs = tf.reshape(mfccs, [1, self.num_frames, self.ceoff, 1])
                input_tensor = mfccs
                
                interpreter = tf.lite.Interpreter(model_path="./{}.tflite".format(uri[0]))
                interpreter.allocate_tensors()

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])

                return output_data
                



if __name__ == '__main__':
	conf = {
		'/': {
			'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
			#'tools.sessions.on': True
		}
	}
	cherrypy.tree.mount(KWSinference(), '/', conf)

	cherrypy.config.update({'server.socket_host': '0.0.0.0'})
	cherrypy.config.update({'server.socket_port': 8080})
	cherrypy.engine.start()
	cherrypy.engine.block()