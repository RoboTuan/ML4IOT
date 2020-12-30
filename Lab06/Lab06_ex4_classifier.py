from DoSomething import DoSomething
import time
import json
import numpy as np
import tensorflow as tf
from datetime import datetime

class thMAE(tf.keras.metrics.Metric):
    def __init__(self, name='thMAE', **kwargs):
        super(thMAE, self).__init__(name=name, **kwargs)
        # We need 2 at least 2 sensors, one to store the MAE computed so far (on the batch)
        # and another variable to store the nmber of batches computed so far, so we can average
        # the final error across the total number of samples processed.
        # I can also write shape=[2] instead of hape=(2, )
        self.total = self.add_weight(name='total', initializer='zeros', shape=(2, ))
        # Shape not needed becasue it's just a scalar value
        self.count = self.add_weight(name='count', initializer='zeros')

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))

        return

    # Set sample_weight=None if I don't need it
    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=0)
        self.total.assign_add(error)
        self.count.assign_add(1)

        return

    def result(self):
        # Computes a safe divide which returns 0 if the y is zero.
        result = tf.math.divide_no_nan(self.total, self.count)

        return result


class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        events = input_json["e"]

        mean = np.array([9.107597, 75.904076])
        std = np.array([8.654227, 16.557089])

        data = np.zeros(shape=(6,2))

        for event in events:

            index = int(event["t"] / 10)

            if event["n"] == "temperature":
                data[index, 0] = float(event["v"])

            else:
                data[index, 1] = float(event["v"])

        data = (data - mean) / (std + 1.e-6)

        data = np.expand_dims(data, axis=0)

        print(data)

        cnn_1d = tf.keras.models.load_model("./Lab06/models/CNN-1D_2/", custom_objects={'thMAE': thMAE})

        logits = cnn_1d.predict(data)
        probs = tf.nn.softmax(logits)
        prob = tf.reduce_max(probs).numpy() * 100

        print("Topic: ", topic, "Prediction: ", logits, "Confidence :", prob)



if __name__ == "__main__":
    test = Subscriber("thClassifier")
    test.run()
    test.myMqttClient.mySubscribe("/276033/th_classifier")

    while True:
        time.sleep(1)

    test.end()