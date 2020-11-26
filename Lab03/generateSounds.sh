#!/bin/bash
# To execute this file do the following:
# chmod +rx generateSounds.sh
# ./generateSounds.sh <number_of_audio_files> <output_directory>
  
# Activate venv
source /home/pi/WORK_DIR/py37/bin/activate

folder=$2
echo Desination folder: $folder

# Good chunk size by the prof is integer division between sample rate and 10
chunks=48000/10
echo Chunks size is: $((chunks))


for (( i = 0 ; i < $1 ; i += 1 )) ; do

  #echo -en 'Type the label of this audio (yes/no): '
  #read label


  python3 /home/pi/WORK_DIR/ML4IOT/Lab03/RecordAudio.py \
  --RecordAudio True \
  --ChunkSize $((chunks)) \
  --output ${folder}/silence${i}.wav \
  --InputFile ${folder}/silence${i}.wav \
  --Result ${folder}/silence${i}.wav

done

# Remove the original files so we only have the resampled ones (stonks)
#rm /home/pi/WORK_DIR/Lab03/${folder}/Silence*

# Deactivate venv
deactivate