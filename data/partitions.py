import os
from typing import *
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split


def load_dataset(
    path: str,
    kern_encoding: str,
    use_distorted_images: bool
    ) -> Tuple[List[str], List[str], List[str]]:
  """Load the dataset from a path as a tuple of lists of strings.
  Discards **kern files that do not have a corresponding image or audio file."""

  extension = '.bekrn' if kern_encoding == 'bekern' else '.krn'

  Y = list(Path(path).rglob('*/*'+extension)) # Get all transcription files
  XA = []
  XI = []

  for y in Y:
    # Get the corresponding images
    if use_distorted_images:
      xi = y.parent / (y.stem + '_distorted.jpg')
    else:
      xi = y.parent / (y.stem + '.jpg')
    # Get the corresponding transcriptions
    
    xa = y.parent / (y.stem + '.wav')

    if xi.exists() and xa.exists():
      XI.append(xi)
      XA.append(xa)
    else:
      print(f'File {y} has no corresponding audio or image. Skipping.')
      # Delete the file from Y
      Y.remove(y)

  return XA, XI, Y


def make_partitions(path: str, kern_encoding: str, use_distorted_images: bool):
  """Make train, validation, and test partitions of the dataset and save them as text files."""

  XA, XI, Y = load_dataset(path, kern_encoding, use_distorted_images)

  # Make 5-fold partitions (only the first will be used)
  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  train_val_idx, test_idx = kf.split(XA).__next__()

  XA_train_val, XA_test = [XA[idx] for idx in train_val_idx], [XA[idx] for idx in test_idx]
  XI_train_val, XI_test = [XI[idx] for idx in train_val_idx], [XI[idx] for idx in test_idx]
  Y_train_val, Y_test = [Y[idx] for idx in train_val_idx], [Y[idx] for idx in test_idx]

  # Split train into train and validation
  XA_train, XA_val, XI_train, XI_val, Y_train, Y_val = train_test_split(XA_train_val, XI_train_val, Y_train_val, test_size=0.2, random_state=42)

  # Save each partition in a file
  partition_folder = Path(path)

  with open(os.path.join(partition_folder, 'train.txt'), 'w') as f:
    for xa, xi, y in zip(XA_train, XI_train, Y_train):
      f.write(f'{xa}\t{xi}\t{y}\n')

  with open(os.path.join(partition_folder, 'val.txt'), 'w') as f:
    for xa, xi, y in zip(XA_val, XI_val, Y_val):
      f.write(f'{xa}\t{xi}\t{y}\n')

  with open(os.path.join(partition_folder, 'test.txt'), 'w') as f:
    for xa, xi, y in zip(XA_test, XI_test, Y_test):
      f.write(f'{xa}\t{xi}\t{y}\n')


def check_and_make_partitions(path: str, kern_encoding: str, use_distorted_images: bool):
  train = Path(path) / 'train.txt'
  test = Path(path) / 'test.txt'
  val = Path(path) / 'val.txt'

  if not train.exists() or not test.exists() or not val.exists():
    print(f'Partitions not found in {path}. Making partitions...')
    make_partitions(path, kern_encoding, use_distorted_images)
  
  return train, val, test


if __name__ == '__main__':
  check_and_make_partitions('data/grandstaff', 'bekern', False)
  check_and_make_partitions('data/grandstaff/beethoven', 'bekern', False)
  check_and_make_partitions('data/grandstaff/chopin', 'bekern', False)
  check_and_make_partitions('data/grandstaff/hummel', 'bekern', False)
  check_and_make_partitions('data/grandstaff/joplin', 'bekern', False)
  check_and_make_partitions('data/grandstaff/mozart', 'bekern', False)
