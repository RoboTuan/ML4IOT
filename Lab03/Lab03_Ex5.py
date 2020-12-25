import argparse
import time as t
from board import D4
import adafruit_dht
from numpy.core.fromnumeric import shape
import tensorflow.lite as tflite
import numpy as np
import pandas as pd
from desktop_scripts.utils import *


def main(freq, per, model):

    #Data acquisition

    readings = [] 
    df = pd.DataFrame(columns=['temperature', 'humidity'])

    print(f"Selected frequencyof measurements: {freq} seconds")
    print(f"Selected period of measurements: {per} seconds")
    print(f"Selected model: {model}")

    dht_device = adafruit_dht.DHT11(D4)


    for _ in range(6):
        temperature = dht_device.temperature
        humidity = dht_device.humidity
        print(f"Temperature: {temperature}, Humidity: {humidity}")
        readings.append((temperature, humidity))
        t.sleep(freq)
    
    inference_data = pd.DataFrame(readings).values.astype(np.float32)
    # print(inference_data.dtype)
    # print(inference_data[0,0].dtype)


    # Inference

    # Mean and std for both columns
    # [ 9.107597 75.904076] [ 8.654227 16.557089]

    mean = [ 9.107597, 75.904076]
    std =  [8.654227, 16.557089]
    input_width = 6
    LABEL_OPTIONS = int(model.split("_")[1])
    #print(LABEL_OPTIONS)

    generator = WindowGenerator(input_width, LABEL_OPTIONS, mean, std)
    inference_ds = generator.make_dataset(inference_data, train=False)


    interpreter = tflite.Interpreter(model_path="/home/pi/WORK_DIR/ML4IOT/Lab03/models/tflite_" + model)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    print("Number of inputs:", len(input_details))
    print("Number of outputs:", len(output_details))
    interpreter.set_tensor(input_details[0]['index'], inference_ds)


        
                   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type=int, default='1', help='Frequency of measurements')
    parser.add_argument('--period', type=int, default='6', help='Period of measurements')
    parser.add_argument('--model', type=str, default='CNN-1D_2', help='Model for inference')

    args, _ = parser.parse_known_args()

    freq = args.frequency
    per = args.period
    model = args.model

    main(freq, per, model)
