import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import *
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='CNN')
parser.add_argument('--mfcc', action='store_true')
parser.add_argument('--silence', action='store_true')
args = parser.parse_args()

MFCC = args.mfcc
MODEL = args.model
SILENCE = args.silence
print(MFCC)

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# zip_path = tf.keras.utils.get_file(
#     origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
#     fname='mini_speech_commands.zip',
#     extract=True,
#     cache_dir='.',
#     cache_subdir='data',
# )

data_dir = os.path.join('.', 'data', 'mini_speech_commands')
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
# Shuffle to have a normal distribution
filenames = tf.random.shuffle(filenames)
#print(filenames)
n = len(filenames)

train_file = filenames[:int(n*0.8)]
val_files = filenames[int(n*0.8):int(n*0.9)]
test_files = filenames[int(n*0.9):]

LABELS = np.array(tf.io.gfile.listdir(str(data_dir)))
#print(LABELS)

LABELS = LABELS[LABELS != "README.md" ]
# Added .DS_Store for mac
LABELS = LABELS[LABELS != ".DS_Store"]
#print(LABELS)

## STFT
    # frame_length = 16000Hz * 0.018ms = 128
    # frame_step = 16000Hz * 0.008ms = 128

## MFCC
    # frame_length = 16000Hz * 0.040ms = 640
    # frame_step = 16000Hz * 0.020ms = 320

STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfcc':False}
MFCC_OPTIONS =  {'frame_length': 640, 'frame_step': 320, 'mfcc':True,
                    'lower_frequency':20, 'upper_frequency':4000,
                    'num_mel_bins':40, 'num_coefficients':10}


if MFCC == True:
    options = MFCC_OPTIONS
    strides = [2,1]
else:
    options = STFT_OPTIONS
    strides = [2,2]

if SILENCE==True:
    num_labels = 9
else:
    num_labels = 8

signal = SignalGenerator(LABELS, 16000, **options)
train_ds = signal.make_dataset(train_file, True)
val_ds = signal.make_dataset(val_files, False)
test_ds = signal.make_dataset(test_files, False)


#print(train_file)

# ds = tf.data.Dataset.from_tensor_slices(train_file)
# print(ds)
# for element in ds:
#     parts = tf.strings.split(element, os.path.sep)
#     print(parts)
#     label = parts[-2]
#     print(label)
#     label_id = tf.argmax(label == LABELS)
#     print(label_id)
#     audio_binary = tf.io.read_file(element)
#     #print(audio_binary)
#     audio, _ = tf.audio.decode_wav(audio_binary)
#     print(audio)
#     audio = tf.squeeze(audio, axis=1)
#     print(audio)

#     zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
#     print(zero_padding)
#     audio = tf.concat([audio, zero_padding], 0)
#     print(audio)
#     audio.set_shape([16000])
#     print(audio)    
#     break

    

if MODEL == "MLP":
    model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu', name='first_dense'),
    keras.layers.Dense(256, activation='relu', name='second_dense'),
    keras.layers.Dense(256, activation='relu', name='third_dense'),
    keras.layers.Dense(num_labels, name='classifier'),
    ])

elif MODEL == "CNN":
    model = keras.Sequential([
        keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides=strides, use_bias=False, name='first_conv'),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides=[1, 1], use_bias=False, name='second_conv'),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides=[1, 1], use_bias=False, name='third_conv'),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.Activation('relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(num_labels, name='classifier')
    ])

elif MODEL == "DS-CNN":
    model = keras.Sequential([
        keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.Activation('relu'),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.Activation('relu'),
        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.Activation('relu'),    
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(num_labels)
    ])

else:
    print("Model not defined")
    sys.exit()



model.compile(optimizer='adam',
              loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

metric = 'val_sparse_categorical_accuracy'

if SILENCE == True:
    checkpoint_filepath = './checkpointSilence/kws_{}_{}/weights'.format(MODEL, MFCC)
else:
    checkpoint_filepath = './checkpoint/kws_{}_{}/weights'.format(MODEL, MFCC)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_sparse_categorical_accuracy', # also metric is ok
    mode='max',
    save_best_only=True)


print("Fit model on training data")

history = model.fit(
    train_ds,
    batch_size=32,
    epochs=20,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(val_ds),
    callbacks=[model_checkpoint_callback]
)

print("Evaluate on test data")
results = model.evaluate(test_ds, verbose=2)

model.summary()

if SILENCE == True:
    save_model_dir = './modelsSilence/kws_{}_{}'.format(MODEL, MFCC)
else:
    save_model_dir = './models/kws_{}_{}'.format(MODEL, MFCC)

if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

# for key in history.history:
#     print(key)

# for element in train_ds:
#     print(element)
#     break 

# y_true = [2, 1]
# y_pred = [[0], [0]]
# m = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
# print(m)
