import joblib
import librosa
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

# JOBLIB MEMORY CACHE:
memory = joblib.Memory('joblib_cache', mmap_mode='r', verbose=0)

@memory.cache
def get_spectrogram_from_file(path, win_length=2048):
  """
  The sampling rate of the input audio files was 22,050Hz, 
  and STFT was calculated with a Hanning window with size 92.88ms 
  (2048 samples) and a hop of 23.22ms (512 samples). Only frequencies 
  between pitches C2 and C7 were considered, extracting 48 bins per octave.
  """

  # Get spectrogram
  y, fs = librosa.load(path, sr=22050)
  stft_fmax = 2093
  stft_frequency_filter_max = librosa.fft_frequencies(sr=fs, n_fft=2048) <= stft_fmax

  stft = librosa.stft(y, hop_length=512, win_length=win_length, window='hann')
  stft = stft[stft_frequency_filter_max]

  stft_db = librosa.amplitude_to_db(np.abs(np.array(stft)), ref=np.max)
  log_stft = ((1./80.) * stft_db) + 1.

  # Normalization
  log_stft = (log_stft - np.amin(log_stft)) / (np.amax(log_stft) - np.amin(log_stft))

  # Convert to PyTorch tensor
  log_stft = torch.from_numpy(log_stft)

  return log_stft


@memory.cache
def get_image_from_file(path, img_height=None):
  x = Image.open(path).convert('L')                       # Convert to grayscale
  if img_height is not None:
      new_width = int(img_height * x.size[0] / x.size[1]) # Get width preserving aspect ratio
      x = x.resize((new_width, img_height))               # Resize
  x = ToTensor()(x)                                       # Convert to tensor (normalizes to [0, 1])
  return x
