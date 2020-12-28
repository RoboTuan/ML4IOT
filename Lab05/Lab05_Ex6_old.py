import cherrypy 
import json
import base64
from cherrypy.process.wspbus import ChannelFailures
import numpy as np
import tensorflow as tf
import sys



class KWSinference(object):
        exposed = True

        def __init__(self):
                self.length = 640
                self.stride = 320
                self.bins = 40
                self.coeff = 10
                self.rate = 16000
                self.resize = 32

                self.num_frames = (self.rate - self.length) // self.stride + 1
                self.num_spectrogram_bins = self.length // 2 + 1

                self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                        self.bins, self.num_spectrogram_bins, self.rate, 20, 4000)
    

        def POST(self, *uri, **query):
        
                print(uri, query)

                audio_string = query.get('audio')
                
                audio_b64bytes = audio_string.encode()
                audio_bytes = base64.decodebytes(audio_b64bytes)
                audio = np.frombuffer(audio_bytes)
                print(audio.shape)
                #sys.exit()

                zero_padding = tf.zeros([self.rate] - tf.shape(audio), dtype=tf.float32)
                audio = tf.concat([audio, zero_padding], 0)
                audio.set_shape([self.rate])


                stft = tf.signal.stft(audio, frame_length=self.length,
                                frame_step=self.stride, fft_length=self.length)
                spectrogram = tf.abs(stft)

                mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
                log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
                mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
                mfccs = mfccs[..., :self.coeff]
                #mfccs = tf.expand_dims(mfccs, -1)
                mfccs = tf.reshape(mfccs, [1, self.num_frames, self.coeff, 1])

                input_tensor = mfccs

                
                LABELS = ['right', 'go', 'no', 'left', 'stop', 'up', 'down', 'yes']
                
                # interpreter = tf.lite.Interpreter(model_path="./{}.tflite".format(uri[0]))
                interpreter = tf.lite.Interpreter(model_path="./dscnn.tflite")
                interpreter.allocate_tensors()

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])

                output_probs = tf.math.softmax(tf.squeeze(output_data, axis=0))
                #print(output_probs)
                probability = float(tf.math.reduce_max(output_probs))
                pred_label = int(tf.math.argmax(output_probs))
                pred_label = LABELS[pred_label]
                print(pred_label, probability)
                #print(LABELS[pred_label])

                output = {
                        'prediction': pred_label,
                        'probability': probability
                }

                output = json.dumps(output)

                return output
                
                # audio_b64bytes = base64.b64encode(b''.join(mfccs))
                # audio_string = audio_b64bytes.decode()


                # output = {
                #         'audio': audio_string
                # }

                # output = json.dumps(output)

                # return output



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