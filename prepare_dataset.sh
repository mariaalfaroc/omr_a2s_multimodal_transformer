#!/bin/bash

############################## IMPORTANT NOTE: READ THIS BEFORE RUNNING THE SCRIPT! ##############################

# Download the GRANDSTAFF dataset from here: https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz
# Put the downloaded files in the data folder (data/grandstaff/...)

# Download a sound font in .sf2 format and update the path in the data/krn2audio.py script

# After that run this script to create the wav files and the partitions

###################################################################################################################

# Extract the files
# mkdir -p data/grandstaff
# tar -xvzf data/grandstaff.tgz -C data/grandstaff
# rm -rf data/grandstaff.tgz

# Parser for the GRANDSTAFF dataset
python -u data/krn2audio.py
python -u data/partitions.py
