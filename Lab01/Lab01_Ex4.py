import argparse
import csv
from datetime import datetime, date, time, timedelta
import time as t
from board import D4
import adafruit_dht


def main(freq, per, out):
    print(f"Selected frequencyof measurements: {freq} seconds")
    print(f"Selected period of measurements: {per} seconds")
    print(f"Output file: {out}")

    readings = []
    dht_device = adafruit_dht.DHT11(D4)

    t_start = datetime.now().replace(microsecond=0)
    t_end = t_start + timedelta(seconds=per)
    
    while datetime.now().replace(microsecond=0) < t_end:
        temperature = dht_device.temperature
        humidity = dht_device.humidity
        print(f"Temperature: {temperature}, Humidity: {humidity}")
        readings.append((datetime.now().replace(microsecond=0), temperature, humidity))
        t.sleep(freq)

    with open(out, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in readings:
            writer.writerow(line)
    
        
                   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type=int, default='5', help='Frequency of measurements')
    parser.add_argument('--period', type=int, default='20', help='Period of measurements')
    parser.add_argument('--output', type=str, default='resultsLab01Ex4.csv', help='Output filename')

    args, _ = parser.parse_known_args()

    freq = args.frequency
    per = args.period
    out = args.output

    main(freq, per, out)
