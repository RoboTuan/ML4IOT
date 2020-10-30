import argparse
import os
import pyaudio
import wave
import time as t

from scipy.io import wavfile
from scipy import signal
import numpy as np


def main(record, rate, seconds, chunk, output, input, resamplingFreq, result):

    # Record audio file
    # A good option can be chunk size = sampling rate // 10
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = rate
    #CHUNK = chunk
    CHUNK = RATE//10
    RECORD_SECONDS = seconds
    WAVE_OUTPUT_FILENAME = output

    if record==True:

        audio = pyaudio.PyAudio()

        print(f"Bith depth: 16")
        print(f"Sample rate: {RATE}")
        print(f"Chunk size: {CHUNK}")
        print(f"Recording time: {RECORD_SECONDS} seconds")
        print(f"Output file name: {WAVE_OUTPUT_FILENAME}")
        print()

        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

        print("recording...")

        frames = []

        start_sensing = t.time()
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        end_sensing = t.time()
        sensing_time = end_sensing - start_sensing

        print("finished recording")
        
        # stop Recording
        stream.stop_stream()
        stream.close()
        # Destroy auio object to destroy memory
        audio.terminate()

        t_start_storing = t.time()

        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        t_end_storing = t.time()

        print(f"time for sensing the audio: {sensing_time}")
        print(f"Time for storing the data on disk: {round(t_end_storing-t_start_storing, 4)} seconds")

        wav_size = os.path.getsize("./file.wav")
        print(f"The size of the wav file is circa: {int(wav_size/1024)} KiloBytes") 
        

    
    # Read the file
    # wavfile from scipy return (rate,audio)
    _, audio = wavfile.read(input)

    audio = signal.resample_poly(audio, 1, RATE/resamplingFreq)

    # Cast the signal to the original datatype (int16)
    audio = audio.astype(np.int16)
    
    storing_start = t.time()
    wavfile.write(result, resamplingFreq, audio)
    storing_end = t.time()

    print(f"Time for storing the resampled audio: {storing_end - storing_start}")
    print(f"Size of rigial audio is circa: {int(os.path.getsize(input)/1024)} KiloBytes")
    print(f"Size of resampled file is circa: {int(os.path.getsize(result)/1024)} KiloBytes")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--RecordAudio', type=bool, default=False, help='Option for recording or using existen file')
    parser.add_argument('--SampleRate', type=int, default=48000, help='Sample rate')
    parser.add_argument('--Seconds', type=int, default=1, help='Amount of time to record in seconds')
    parser.add_argument('--ChunkSize', type=int, default=1024, help='Size of the different chunks of audio')
    parser.add_argument('--output', type=str, default='file.wav', help='Name of the output wav file')
    parser.add_argument('--InputFile', type=str, default='file.wav', help='Input file for the audio processing')
    parser.add_argument('--ResamplingRatio', type=int, default=16000, help='Resampling sampling ratio')
    parser.add_argument('--Result', type=str, default='resampled_file.wav', help='Audio file after resampling')

    args, _ = parser.parse_known_args()

    record = args.RecordAudio
    rate = args.SampleRate
    seconds = args.Seconds
    chunk = args.ChunkSize
    output = args.output
    input = args.InputFile
    resamplingFreq = args.ResamplingRatio
    result = args.Result

    main(record, rate, seconds, chunk, output, input, resamplingFreq, result)
