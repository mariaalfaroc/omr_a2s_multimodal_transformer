from pathlib import Path
from torch.utils.data import Dataset
from typing import *
import os
import torch

from .partitions import check_and_make_partitions
from .KrnConverter import KrnConverter, ENCODING_OPTIONS
from .utils import get_spectrogram_from_file, get_image_from_file

NUM_CHANNELS = 1
IMG_HEIGHT = 256
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
    self.Y = [krn_encoder.encode(file) for file in self.Y_files]
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
    self.w2i[PAD_TOKEN] = 0
    self.i2w[0] = PAD_TOKEN

    # TODO Save dictionaries
    #np.save(W2I_PATH, w2i)
    #np.save(I2W_PATH, i2w)

  def __len__(self) -> int:
    return len(self.XA)
  
  def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Get spectrogram
    xa = get_spectrogram_from_file(self.XA[index])

    # Get image
    xi = get_image_from_file(self.XI[index])

    # Get transcript
    y = self.Y[index]
    # Convert to indices
    y = [self.w2i[token] for token in y]
    # Convert to PyTorch tensor
    y = torch.tensor(y)
    
    return xa, xi, y
  
  def get_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
    return self.w2i, self.i2w
  
  def get_files(self, index) -> Tuple[str, str, str]:
    return self.XA[index], self.XI[index], self.Y_files[index]


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


def pad_batch_images(X):
    #max_width = max(x, key=lambda sample: sample.shape[2]).shape[2]
    #x = torch.stack([F.pad(i, pad=(0, max_width - i.shape[2])) for i in x], dim=0)
    return [x.long() for x in X]


def pad_batch_transcripts(X):
    #max_length = max(x, key=lambda sample: sample.shape[0]).shape[0]
    #x = torch.stack([F.pad(i, pad=(0, max_length - i.shape[0])) for i in x], dim=0)
    x = [x.long() for x in X]
    return x


def batch_preparation(batch):
    xa, xi, y = zip(*batch)
    # # Zero-pad spectrograms/images to maximum batch spectrogram/image width
    # xa = pad_batch_images(xa)
    # xi = pad_batch_images(xi)
    # # Decoder input: <sot> symbols
    # dec_in = pad_batch_transcripts(y[:-1])
    # # Decoder output: symbols <eot>
    # dec_out = pad_batch_transcripts(y[1:])
    return xa, xi, y, y


###################################################################### PYTORCH DATALOADER UTILS:


if __name__ == '__main__':
  import argparse

  from torch.utils.data import DataLoader
  from torchvision.utils import make_grid, save_image

  CHECK_DIR = "check"
  if not os.path.isdir(CHECK_DIR):
    os.mkdir(CHECK_DIR)

  parser = argparse.ArgumentParser()
  parser.add_argument('--use_distorted_images', action='store_true', help='Use distorted images')
  parser.add_argument('--fold_id', type=int, default=1, choices=[1, 2, 3, 4, 5], help='Fold id')
  parser.add_argument('--kern_encoding', type=str, default='bekern', choices=ENCODING_OPTIONS, help='Kern encoding')
  #parser.add_argument('--keep_ligatures', action='store_true', help='Keep ligatures')
  args = parser.parse_args()

  
  train_dataset, val_dataset, test_dataset = load_gs_datasets(path='data/grandstaff/mozart',
                                                              kern_encoding=args.kern_encoding,
                                                              use_distorted_images=args.use_distorted_images)
  
  train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=batch_preparation)

  print(f'Train dataset size: {len(train_dataset)}')
  for xa, xi, dec_in, dec_out in train_loader:
    print('Types:')
    print('\txa:', xa[0].dtype)
    print('\txi:', xi[0].dtype)
    print('\tdec_in:', dec_in[0].dtype)
    print('\tdec_out:', dec_out[0].dtype)
    print('Shapes:')
    print('\txa:', xa[0].shape)
    print('\txi:', xi[0].shape)
    print('\tdec_in:', dec_in[0].shape)
    print('\tdec_out:', dec_out[0].shape)

    # Save batch spectrogram/images
    save_image(make_grid(list(xa), nrow=4), f'{CHECK_DIR}/xa_train_batch.jpg')
    save_image(make_grid(list(xi), nrow=4), f'{CHECK_DIR}/xi_train_batch.jpg')

    # See first sample
    w2i, i2w = train_dataset.get_vocabulary()
    print(f'Shape with padding: {dec_in[0].shape}')
    print('Decoder input:', [i2w[i.item()] for i in dec_in[0]])
    print('Decoder output:', [i2w[i.item()] for i in dec_out[0]])
    save_image(xi[0], f'{CHECK_DIR}/xi0_train_batch.jpg')

    break
