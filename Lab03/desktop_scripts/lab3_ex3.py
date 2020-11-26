import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from utils import *
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model name')
parser.add_argument('--labels', type=int, required=True, help='model output')
args = parser.parse_args()


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Use the get_file method from tf.keras.utils
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    # Extrac the file since it's a zipped
    extract=True,
    cache_dir='.', cache_subdir='data')

# Split the path name into a pair root and ext so that I then can load the 
csv_path, _ = os.path.splitext(zip_path)
# Load the dataset using the read_csv method from pandas
df = pd.read_csv(csv_path)

# Select the temperature and humidity columns (#2 and #5).
column_indices = [2, 5]
columns = df.columns[column_indices]
# Convert the data into a 32-bit float numpy array
data = df[columns].values.astype(np.float32)

# Split the data into three different sets: train (70%), validation (10%), test (20%)
n = len(data)
train_data = data[0:int(n*0.7)]
print(train_data.shape)
# print(train_data)
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

# Mean and std for both columns
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
#print(mean, std)

# Length of the window (6x2)
input_width = 6
LABEL_OPTIONS = args.labels

# Create datasets
generator = WindowGenerator(input_width, LABEL_OPTIONS, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

MODEL_OPTIONS = args.model


if LABEL_OPTIONS < 2:
    units=1
elif LABEL_OPTIONS == 2:
    units=2
else:
    print(f"Label option must be less than 2")
    sys.exit()

if MODEL_OPTIONS == "MLP":
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(6, 2)),
    keras.layers.Dense(128, activation='relu', name='first_dense'),
    keras.layers.Dense(128, activation='relu', name='second_dense'),
    keras.layers.Dense(units, name='third_dense')
    ])

elif MODEL_OPTIONS == "CNN-1D":
    model = keras.Sequential([
        keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', name='first_conv'),
        keras.layers.Flatten(input_shape=(64,)),
        keras.layers.Dense(units=64, activation='relu', name='first_dense'),
        keras.layers.Dense(units, name='second_dense')
    ])

elif MODEL_OPTIONS == "LSTM":
    model = keras.Sequential([
        keras.layers.LSTM(64),
        keras.layers.Flatten(),
        keras.layers.Dense(units)
    ])

else:
    print("Invalid model selected")
    sys.exit()

# model.build((32,6,2))
# model.summary()



if LABEL_OPTIONS <2:
    model.compile(optimizer='adam',
                loss=[tf.keras.losses.MeanSquaredError()],
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

    print("Fit model on training data")

    history = model.fit(
        train_ds,
        batch_size=32,
        epochs=2,
        #epochs=20,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(val_ds)
    )

    print("Evaluate on test data")
    results = model.evaluate(test_ds, verbose=2)

    model.summary()
    #print("test loss, test MAE:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    # print("Generate predictions for 3 samples")
    # predictions = model.predict(test_ds[:3])
    # print("predictions shape:", predictions.shape)

else:
    model.compile(optimizer='adam',
                loss=[tf.keras.losses.MeanSquaredError()],
                metrics=[thMAE()])

    print("Fit model on training data")

    history = model.fit(
        train_ds,
        batch_size=32,
        #epochs=2,
        epochs=20,
        validation_data=(val_ds)
    )

    print("Evaluate on test data")
    loss, error = model.evaluate(test_ds, verbose=2)
    print(error)

    model.summary()



