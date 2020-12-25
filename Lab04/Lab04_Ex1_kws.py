import argparse
import numpy as np
import tensorflow as tf
import sys
import zlib


parser = argparse.ArgumentParser()
# In model put model_True/False, like MLP_False
parser.add_argument('--model', type=str, default='MLP')
parser.add_argument('--mfcc', action='store_true')
parser.add_argument('--silence', action='store_true')
parser.add_argument('--ptq', type=str, default='NoPTQ')
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--magPrun', action='store_true')
args = parser.parse_args()

if args.mfcc == True:
    dataset_name = './kws_True_test_silence/'
    tensor_specs = (tf.TensorSpec([None, 49, 10, 1], dtype=tf.float32),
                    tf.TensorSpec([None], dtype=tf.int64))
else:
    dataset_name = './kws_False_test_silence/'
    tensor_specs = (tf.TensorSpec([None, 32, 32, 1], dtype=tf.float32),
                    tf.TensorSpec([None], dtype=tf.int64))

test_ds = tf.data.experimental.load(dataset_name, tensor_specs)
test_ds = test_ds.unbatch().batch(1)




# put tflite_kws_Model_False/True_ptq
if args.silence == True:

    model_path='./modelsSilence/tflite_kws_{}_{}_{}_{}_{}'.format(args.model, args.mfcc, args.ptq, args.alpha, args.magPrun)

    # if args.magPrun == True:
    #     with open(model_path, 'wb') as fp:
    #         tflite_decompressed = zlib.decompress(fp)
    #         fp.write(tflite_decompressed)

    interpreter = tf.lite.Interpreter(model_path=model_path)

else:

    model_path='./models/tflite_kws_{}_{}_{}_{}_{}'.format(args.model, args.mfcc, args.ptq, args.alpha, args.magPrun)

    # if args.magPrun == True:
    #     with open(model_path, 'wb') as fp:
    #         tflite_decompressed = zlib.decompress(fp)
    #         fp.write(tflite_decompressed) 

    interpreter = tf.lite.Interpreter(model_path=model_path)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

accuracy = 0
count = 0
for x, y_true in test_ds:
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    y_pred = y_pred.squeeze()
    y_pred = np.argmax(y_pred)
    y_true = y_true.numpy().squeeze()
    accuracy += y_pred == y_true
    count += 1

accuracy /= float(count)
print('Accuracy {:.2f}'.format(accuracy*100))
