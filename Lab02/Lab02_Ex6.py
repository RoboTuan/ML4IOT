import tensorflow as tf
import time as t
import os

# Read the audio signal from Ex. 5 with the tf.io.read_file method
resampled_file = "./resampled_file.wav"
audio = tf.io.read_file(resampled_file)

# Convert the signal in a TensorFlow tensor using tf.audio.decode_wav method
tf_audio, rate = tf.audio.decode_wav(audio)
tf_audio = tf.squeeze(tf_audio, 1)

# Convert the waveform in a spectrogram applying the STFT
frame_length = int(16000 * 0.04)
frame_step = int(16000 * 0.02)

spectogram_start = t.time()
stft = tf.signal.stft(tf_audio, frame_length=frame_length,
                        frame_step=frame_step,
                        fft_length=frame_length)

spectrogram = tf.abs(stft)
spectrogram_end = t.time()

print(f"Execution time needed to compute the spectrogram {spectrogram_end-spectogram_start} seconds")

# Store the spectrogram and measure the file size
serialized_spectogram = tf.io.serialize_tensor(spectrogram)
file = "serialized_spectogram"
tf.io.write_file(file, serialized_spectogram)
print(f"Size of resampled file is circa: {int(os.path.getsize(resampled_file)/1024)} KiloBytes")
print(f"Size of the serialized spectogram is circa: {int(os.path.getsize(file)/1024)} KiloBytes")


# Now we have to generate the yes/no sequence of files
# Use the generateSounds.sh and the instruction in it to do so
# The files will be in the audio_sequence directory.
# Now iterate though them and do the 4.6 part of this exercise

folder = './audio_sequence/'
print("------------------------------------------")

for filename in os.listdir(folder):

    print(f"File to process: {filename}")

    # This is like the previous section, but is done for each file
    # in the audio_sequence directory
    audio = tf.io.read_file(folder + filename)

    tf_audio, rate = tf.audio.decode_wav(audio)
    tf_audio = tf.squeeze(tf_audio, 1)

    frame_length = int(16000 * 0.04)
    frame_step = int(16000 * 0.02)

    stft = tf.signal.stft(tf_audio, frame_length=frame_length,
                        frame_step=frame_step,
                        fft_length=frame_length)
    
    spectrogram = tf.abs(stft)

    # Transpose the spectrogram to represent time on x-axis
    image = tf.transpose(spectrogram)

    # Add the “channel” dimension
    image = tf.expand_dims(image, -1)

    # Take the logarithm of the spectrogram for better visualization
    image = tf.math.log(image + 1.e-6)

    # Apply min/max normalization and multiply by 255 (images are unsigned bytes)
    min_ = tf.reduce_min(image)
    max_ = tf.reduce_max(image)
    image = (image - min_) / (max_ - min_)
    image = image * 255.

    # Cast the tensor to uint8:
    image = tf.cast(image, tf.uint8)

    # Convert the tensor to a PNG 
    image = tf.io.encode_png(image)

    new_file = './STFTs/' + filename.split(".")[0] + ".png"
    tf.io.write_file(new_file, image)










