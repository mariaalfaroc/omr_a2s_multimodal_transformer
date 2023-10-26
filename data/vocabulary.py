import os

import numpy as np
from loguru import logger
from pathlib import Path

@logger.catch
def check_and_retrieve_vocabulary(YSequences, path, padding_token):
  # make path from path and name of vocabulary

  w2i_path = Path(path).joinpath("w2i.npy")
  i2w_path = Path(path).joinpath("i2w.npy")

  w2i = {}
  i2w = {}

  # make all subdirectories in path if they don't exist
  if not os.path.isdir(path):
    os.makedirs(path)
  
  if os.path.isfile(w2i_path):
    w2i = np.load(w2i_path, allow_pickle=True).item()
    i2w = np.load(i2w_path, allow_pickle=True).item()
  else:
    w2i, i2w = make_vocabulary(YSequences, w2i_path, i2w_path, padding_token)

  return w2i, i2w

def make_vocabulary(YSequences, w2i_path, i2w_path, padding_token):
  vocabulary = set()
  for samples in YSequences:
    for element in samples:
      vocabulary.update(element)

  # Create vocabulary
  w2i = {symbol:idx+1 for idx, symbol in enumerate(vocabulary)}
  i2w = {idx+1:symbol for idx, symbol in enumerate(vocabulary)}
  
  w2i[padding_token] = 0
  i2w[0] = padding_token

  # Save the vocabulary
  np.save(w2i_path, w2i)
  np.save(i2w_path, i2w)

  return w2i, i2w
