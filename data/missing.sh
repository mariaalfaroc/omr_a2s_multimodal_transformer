#!/bin/bash

cd ~/DATASETS/grandstaff
find . -name *.krn | while read krnfile; do
    # replace extension with .wav
    wavfile=${krnfile%.krn}.wav
    # check if file exists
    if [ ! -f ${wavfile} ]; then
        echo $krnfile
    fi
done
