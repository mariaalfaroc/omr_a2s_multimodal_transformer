#!/bin/bash

############################## IMPORTANT NOTE: READ THIS BEFORE RUNNING THE SCRIPT! ##############################

# Download the GRANDSTAFF dataset from here: https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz
# Put the downloaded files in the data folder
# After that run the prepare_dataset.sh script in the data

###################################################################################################################

# Extract the files
# mkdir -p data/grandstaff
# tar -xvzf data/grandstaff.tgz -C data/grandstaff
# rm -rf data/grandstaff.tgz

# Parser for the GRANDSTAFF dataset
python -u data/krn2audio.py
