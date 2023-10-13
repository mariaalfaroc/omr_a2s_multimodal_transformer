import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, joblib
from pathlib import Path
from typing import *

import torch
import librosa
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, train_test_split

from data.encodingGS import krnConverter
from vocab.preprocessing import EOT_TOKEN, SOT_TOKEN, CON_TOKEN, COC_TOKEN, COR_TOKEN



def encodeKrnSeq(in_seq):

    out_seq = list()

    # Start of sequence:
    out_seq.append(SOT_TOKEN)

    # For every single time step:
    for step in in_seq:
        encoded_line = list()
        # For every single voice in the time step:
        for it_voice in range(len(step)-1):
            elements_in_voice = step[it_voice].split()
            # All notes in a voice except for the last one:
            for it_element in range(len(elements_in_voice)-1):
                encoded_line.append(elements_in_voice[it_element])
                encoded_line.append(CON_TOKEN) # Change of note
            
            # Last note in the current voice
            encoded_line.append(elements_in_voice[-1])
            encoded_line.append(COC_TOKEN) # Change of voice

        # All notes in the last voice except for the last one:
        elements_in_voice = step[-1].split()
        for it_element in range(len(elements_in_voice)-1):
            encoded_line.append(elements_in_voice[it_element])
            encoded_line.append(CON_TOKEN) # Change of note
        # Last note in the current voice
        encoded_line.append(elements_in_voice[-1])
        encoded_line.append(COR_TOKEN) # Change of voice

        out_seq.extend(encoded_line)

    out_seq[-1] = EOT_TOKEN
    return out_seq




if __name__ == '__main__':
    convKRN = krnConverter()

    path = 'data/grandstaff/chopin/mazurkas/mazurka33-3/maj2_up_m-15-18.bekrn'

    res = convKRN.cleanKernFile(path)
    encoded_seq = encodeKrnSeq(res)
    print(encoded_seq)


    # ## Checking all files:
    # base_path = 'data/'
    # with open('dst.txt', 'w') as fout:
    #     for root, dir_names, file_names in os.walk(base_path):
    #         for single_file in file_names:
    #             if single_file.endswith('bekrn'):
    #                 target_file = os.path.join(root, single_file)
    #                 try:
    #                     res = convKRN.cleanKernFile(target_file)
    #                     encoded_seq = encodeKrnSeq(res)
    #                     fout.write("{} - {}\n".format(target_file, "Done!"))
    #                 except:
    #                     fout.write("{} - {}\n".format(target_file, "Fail!"))