import os

import numpy as np

from data.preprocessing import load_grandstaff_dataset
from data.encodingGS import krnConverter


###################################################################### SPECIAL TOKENS:


PAD_TOKEN = '<pad>' # Padding token
EOT_TOKEN = '<eot>' # End-of-transcript token
SOT_TOKEN = '<sot>' # Start-of-transcript token
CON_TOKEN = '<con>' # Change-of-note (change-of-note) token
COC_TOKEN = '<coc>' # Change-of-column (change-of-voice) token
COR_TOKEN = '<cor>' # Change-of-row (change-of-event) token


###################################################################### VOCABULARY UTILS:


W2I_PATH = 'vocab/w2i.npy'
I2W_PATH = 'vocab/i2w.npy'


def check_and_retrieve_vocabulary(kern_encoding: str = 'bekern', keep_ligatures: bool = True):
    w2i = {}
    i2w = {}

    if os.path.isfile(W2I_PATH):
        w2i = np.load(W2I_PATH, allow_pickle=True).item()
        i2w = np.load(I2W_PATH, allow_pickle=True).item()
    else:
        # Load dataset
        _, _, Y = load_grandstaff_dataset() 
        w2i, i2w = make_vocabulary(Y, kern_encoding, keep_ligatures)

    return w2i, i2w


def make_vocabulary(Y, kern_encoding: str = 'bekern', keep_ligatures: bool = True):
    vocab = set()

    krn_encoder = krnConverter(kern_encoding, keep_ligatures)

    for y in Y:
        y_clean = krn_encoder.convert(y)
        y_clean = ''.join(y_clean)
        y_clean = y_clean.replace('\n', ' ')
        y_clean = y_clean.replace('\t', ' ')
        y_clean = y_clean.split(' ')
        vocab.update(y_clean)

    # Create dictionaries
    # Add special tokens and reserve index 0 for padding
    vocab.update([EOT_TOKEN, SOT_TOKEN, COC_TOKEN, COR_TOKEN])
    w2i = {w: i+1 for i, w in enumerate(vocab)}
    i2w = {i+1: w for i, w in enumerate(vocab)}
    w2i[PAD_TOKEN] = 0
    i2w[0] = PAD_TOKEN

    # Save dictionaries
    np.save(W2I_PATH, w2i)
    np.save(I2W_PATH, i2w)

    return w2i, i2w