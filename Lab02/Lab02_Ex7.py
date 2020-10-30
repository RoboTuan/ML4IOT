import tensorflow as tf
import os
import time as t


# Read the byte string stored on disk
spectrogram_byteString = tf.io.read_file('./serialized_spectogram')

# Convert the byte string in a TF tensor
spectrogram = tf.io.parse_tensor(spectrogram_byteString, out_type=tf.float32)

# Compute the log-scaled Mel spectrogram with 40 Mel bins, 
# 20Hz as lower frequency, and 4kHz as upper frequency
lower_frequency = 20
upper_frequency = 4000
num_mel_bins = 40
sampling_rate = 16000

num_spectrogram_bins = spectrogram.shape[-1]
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    num_mel_bins,
                    num_spectrogram_bins,
                    sampling_rate, # 16000
                    lower_frequency,
                    upper_frequency)

mel_spectrogram = tf.tensordot( spectrogram,
                  linear_to_mel_weight_matrix,
                  1) 
mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

# Compute the MFCCs from the log-scaled Mel spectrogram and take the first 10 coefficients
mfccs = tf.signal.mfccs_from_log_mel_spectrograms( log_mel_spectrogram)[..., :10]

# Store the MFCCs (in binary file) and measure the file size
serialized_mfccs = tf.io.serialize_tensor(spectrogram)
file = "./serialized_mfccs"
tf.io.write_file(file, serialized_mfccs)
print(f"Size of the serialized mfccs is circa: {int(os.path.getsize(file)/1024)} KiloBytes")


folder = './audio_sequence/'
print("------------------------------------------")
# Compute the MFCCs of the “yes” and “no” samples

for filename in os.listdir(folder):
    print(f"File to process: {filename}")

    audio = tf.io.read_file(folder + filename)

    # Convert the signal in a TensorFlow tensor using tf.audio.decode_wav method
    tf_audio, rate = tf.audio.decode_wav(audio)
    tf_audio = tf.squeeze(tf_audio, 1)

    # Convert the waveform in a spectrogram applying the STFT
    frame_length = int(16000 * 0.04)
    frame_step = int(16000 * 0.02)

    stft = tf.signal.stft(tf_audio, frame_length=frame_length,
                            frame_step=frame_step,
                            fft_length=frame_length)

    # this has already a dtype=float32
    spectrogram = tf.abs(stft)

    num_spectrogram_bins = spectrogram.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                        num_mel_bins,
                        num_spectrogram_bins,
                        sampling_rate,
                        lower_frequency,
                        upper_frequency)

    mel_spectrogram = tf.tensordot( spectrogram,
                        linear_to_mel_weight_matrix,
                        1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms( log_mel_spectrogram)[..., :10]

    # Transpose the spectrogram to represent time on x-axis
    # TODO: see if it's necessary
    image = tf.transpose(mfccs)

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

    new_file = './MFCCs/' + filename.split(".")[0] + ".png"
    tf.io.write_file(new_file, image)


   



