import tensorflow as tf
import numpy as np
import tensorflow.lite as tflite

x_train = [-1, 0, 1]
y_train = [-3, -1, 1]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
model.summary()
opt = tf.keras.optimizers.SGD(learning_rate=0.05)
model.compile(optimizer=opt, loss='mean_squared_error')
model.fit(x=x_train, y=y_train, epochs=30)

model.save("my_model")

converter = tf.lite.TFLiteConverter.from_saved_model("my_model")
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print(tflite_model)

with open('my_model.tflite', 'wb') as f:
    f.write(tflite_model)


interpreter = tflite.Interpreter(model_path="my_model.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Number of inputs:", len(input_details))
print("Number of outputs:", len(output_details))
print("Input name:", input_details[0]['name'])
print("Input shape:", input_details[0]['shape'])

inputs = []
outputs = []
for i in range(10):

    my_input = np.array(np.random.uniform(-1, 1, input_details[0]['shape']), dtype=np.float32)
    print("Input:", my_input)
    inputs.append(my_input[0, 0])
    
    interpreter.set_tensor(input_details[0]['index'], my_input)
    print("yolo")
    
    interpreter.invoke()
    
    my_output = interpreter.get_tensor(output_details[0]['index'])
    print("Output:", my_output)  
    outputs.append(my_output[0, 0])