from pathlib import Path
from torch.utils.data import Dataset
from typing import *
import os
import torch
import numpy as np

from .partitions import check_and_make_partitions
from .utils import get_spectrogram_from_file, get_image_from_file

from utils.kern import KrnConverter, ENCODING_OPTIONS

NUM_CHANNELS = 1
SPECTROGRAM_HEIGHT = 195
SCORE_HEIGHT = 256
TEACHER_FORCING_ERROR_RATE = 0.2
PAD_TOKEN_INDEX = 0

EOT_TOKEN = '<eot>' # End-of-transcription token
SOT_TOKEN = '<sot>' # Start-of-transcription token
PAD_TOKEN = '<pad>' # Padding token

class GrandStaffDataset(Dataset):
  def __init__(
      self,
      path: str,
      w2i: Dict[str, int] = None,
      i2w: Dict[int, str] = None,
      use_distorted_images: bool = False,
      kern_encoding: str = 'bekern',
      keep_ligatures: bool = True) -> None:
    
    # Kern Converter
    krn_encoder = KrnConverter(kern_encoding, keep_ligatures)

    self.XA, self.XI, self.Y_files = self.__load_files__(path)
    self.Y = [[SOT_TOKEN] + krn_encoder.encode(file) + [EOT_TOKEN] for file in self.Y_files]
    self.__make_vocabulary__()
  
  def __load_files__(self, path: str) -> Tuple[List[str], List[str], List[str]]:
    """Load a partition from a text file."""
    XA, XI, Y = [], [], []
    with open(path, 'r') as f:
      for line in f:
        line = line.strip()
        xa, xi, y = line.split('\t')
        XA.append(xa)
        XI.append(xi)
        Y.append(y)  
    return XA, XI, Y

  def __make_vocabulary__(self) -> None:
    vocab = set()

    for y in self.Y:
      vocab.update(y)

    # Add special tokens ! no longer needed, they are already in the transcriptions
    # vocab.update([EOT_TOKEN, SOT_TOKEN, CON_TOKEN, COC_TOKEN, COR_TOKEN])

    # Create dictionaries and reserve index 0 for padding
    self.w2i = {w: i+1 for i, w in enumerate(vocab)}
    self.i2w = {i+1: w for i, w in enumerate(vocab)}
    self.w2i[PAD_TOKEN] = PAD_TOKEN_INDEX
    self.i2w[PAD_TOKEN_INDEX] = PAD_TOKEN

    # TODO Save dictionaries
    #np.save(W2I_PATH, w2i)
    #np.save(I2W_PATH, i2w)

  def __len__(self) -> int:
    return len(self.XA)

  def __apply_teacher_forcing__(self, sequence: List[int]) -> List[int]:
    errored_sequence = sequence.copy()
    for token in range(1, len(sequence)):
      if np.random.rand() < TEACHER_FORCING_ERROR_RATE and sequence[token] != PAD_TOKEN_INDEX:
        errored_sequence[token] = np.random.randint(0, len(self.w2i))
    return errored_sequence
      
  def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Get spectrogram
    xa = get_spectrogram_from_file(self.XA[index])

    # Get image
    xi = get_image_from_file(self.XI[index])

    # Get transcript
    y = self.Y[index]
    # Convert to indices
    y = [self.w2i[token] for token in y]
    # Create decoder input
    decoder_input = self.__apply_teacher_forcing__(y) # Remove last token (EOT)
    # Convert to PyTorch tensor
    y = torch.tensor(y)
    decoder_input = torch.tensor(decoder_input)
    
    return xa, xi, decoder_input, y
  
  def get_dictionaries(self) -> Tuple[Dict[str, int], Dict[int, str]]:
    return self.w2i, self.i2w
  
  def get_files(self, index) -> Tuple[str, str, str]:
    return self.XA[index], self.XI[index], self.Y_files[index]

  def get_max_audio_hw(self):
    shapes = [get_spectrogram_from_file(file).shape for file in self.XA]
    m_height, m_width = np.max(shapes, axis=0)
    return m_height, m_width

  def get_max_score_hw(self):
    m_height = np.max([img.shape[0] for img in self.XI])
    m_width = np.max([img.shape[1] for img in self.XI])
    return m_height, m_width

  def get_max_seqlen(self):
    return max([len(seq) for seq in self.Y])

  def get_vocab_size(self):
    return len(self.w2i)

  def get_gt(self):
    return self.Y


###################################################################### DATALOADER FUNCTION:


def load_gs_datasets(path: str, kern_encoding: str, use_distorted_images: False):
  assert os.path.exists(path), f'Path {path} does not exist.'
  assert kern_encoding in ENCODING_OPTIONS, f'You must chose one of the possible encoding options: {",".join(ENCODING_OPTIONS)}'
  
  train, val, test = check_and_make_partitions(path, kern_encoding, use_distorted_images)
  
  train_dataset = GrandStaffDataset(train, kern_encoding=kern_encoding, use_distorted_images=use_distorted_images)
  val_dataset = GrandStaffDataset(val, kern_encoding=kern_encoding, use_distorted_images=use_distorted_images)
  test_dataset = GrandStaffDataset(test, kern_encoding=kern_encoding, use_distorted_images=use_distorted_images)

  return train_dataset, val_dataset, test_dataset


###################################################################### PYTORCH DATALOADER UTILS:

import torch.nn.functional as F

def pad_batch_images(X, pad_value=0.):
  if X[0].dim() == 2:
    X = [i.unsqueeze(0) for i in X] # Add channel dimension to spectrograms
  #widths = [i.shape[2] for i in X]
  max_width = max([i.shape[2] for i in X])
  #print(f'max_width={max_width}, widths={widths}')
  # Pad images to maximum batch image width
  X = torch.stack([F.pad(i, value=pad_value, pad=(0, max_width - i.shape[2])) for i in X], dim=0)
  return X


def pad_batch_transcripts(X):
  #widths = [x.shape[0] for x in X]
  max_length = max(X, key=lambda sample: sample.shape[0]).shape[0]
  #print(f'max_length={max_length}, widths={widths}')
  # Pad transcripts to maximum batch transcript length using padding token (0 = <pad>)
  X = torch.stack([F.pad(x, value=0, pad=(0, max_length - x.shape[0])) for x in X], dim=0)
  #X = [x.long() for x in X]
  return X


def batch_preparation(batch):
  XA, XI, Y, GT = zip(*batch)
  # # Zero-pad spectrograms/images to maximum batch spectrogram/image width
  XA = pad_batch_images(XA, pad_value=0.) # background for spectrograms is black
  XI = pad_batch_images(XI, pad_value=1.) # background for scores is white
  # Decoder input: <sot> symbols
  # remove last token from all Y elements
  Y = pad_batch_transcripts([y[:-1] for y in Y]) # Remove <eot> from decoder input  
  # Decoder output: symbols <eot>
  GT = pad_batch_transcripts([y[1:] for y in GT]) # Remove <sot> from decoder output
  #return XA, XI, Y, GT
  return XA, Y.long(), GT.long()
