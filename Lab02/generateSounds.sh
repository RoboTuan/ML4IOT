#!/bin/bash
# To execute this file do the following:
# chmod +rx generateSounds.sh
# ./generateSounds.sh <number_of_audio_files>
  
# Activate venv
source /home/pi/WORK_DIR/py37/bin/activate 

folder='./audio_sequence/'
echo Desination folder: $folder

# Good chunk size by the prof is integer division between sample rate and 10
chunks=48000/10
echo Chunks size is: $((chunks))


for (( i = 1 ; i <= $1 ; i += 1 )) ; do

  echo -en 'Type the label of this audio (yes/no): '
  read label

  python3 ./Lab02_Ex5.py \
  --RecordAudio True \
  --ChunkSize $((chunks)) \
  --output ${folder}file${i}.wav \
  --InputFile ${folder}file${i}.wav \
  --Result ${folder}resampled_file${i}_${label}.wav

done

# Remove the original files so we only have the resampled ones (stonks)
rm /home/pi/WORK_DIR/Lab02/${folder}/file*

# Deactivate venv
deactivate