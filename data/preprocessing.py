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

from encoding import krnConverter
from vocab.preprocessing import check_and_retrieve_vocabulary
from vocab.preprocessing import EOT_TOKEN, SOT_TOKEN, COC_TOKEN, COR_TOKEN

# TODO:
# - Poner la cache en RAID si se lanza en la DGX
# - Decidir el tamaño del alto de imagen a usar
# - Teacher forcing para el input del decoder (o se implementa en el modelo o en el dataset)
# - Comprobar que no hay errores en la carga (he dejado un script de prueba en el main de este fichero)


# JOBLIB MEMORY CACHE:
memory = joblib.Memory('cache', mmap_mode='r', verbose=0)


def load_fold_data(
        use_distorted_images: bool = True,
        fold_id: int = 1,
        kern_encoding: str = 'kern',
        keep_ligatures: bool = True):
    
    # Retrieve partitions
    partitions = check_and_retrieve_partitions(use_distorted_images, fold_id)
    # Retrieve vocabulary
    w2i, i2w = check_and_retrieve_vocabulary(kern_encoding, keep_ligatures)
    return partitions, w2i, i2w


###################################################################### LOAD UTILS:


PARTITIONS_FOLDER_PATH = 'data/partitions'


def check_and_retrieve_partitions(use_distorted_images: bool = True, fold_id: int = 1):
    # fold_id = 1, 2, 3, 4, 5
    # Check if partitions exist
    if not os.path.isdir(PARTITIONS_FOLDER_PATH):
        # Load dataset
        XA, XI, Y = load_grandstaff_dataset() # Use the same splits for original and distorted images
        # Make partitions
        make_partitions(XA, XI, Y)
    
    # Retrieve partitions
    partition_folder = os.path.join(PARTITIONS_FOLDER_PATH, f'fold_{fold_id}')
    XATrain, XITrain, YTrain = load_partition(os.path.join(partition_folder, 'train.txt'), use_distorted_images)
    XAVal, XIVal, YVal = load_partition(os.path.join(partition_folder, 'val.txt'), use_distorted_images)
    XATest, XITest, YTest = load_partition(os.path.join(partition_folder, 'test.txt'), use_distorted_images)

    return {
        'train': (XATrain, XITrain, YTrain),
        'val': (XAVal, XIVal, YVal),
        'test': (XATest, XITest, YTest),
    }


def load_grandstaff_dataset(use_distorted_images: bool = True):
    XA = list(Path('.').rglob('*/*.wav')) # Get all wav files
    XI = []
    Y = []

    for xa in XA:
        # Get the corresponding images
        if use_distorted_images:
            xi = xa.parent / (xa.stem + '_distorted.jpg')
        else:
            xi = xa.parent / (xa.stem + '.jpg')
        # Get the corresponding transcriptions
        y = xa.parent / (xa.stem + '.krn')

        if xi.exists() and y.exists():
            XI.append(xi)
            Y.append(y)
        else:
            print(f'File {xa} has no corresponding image or transcription. Skipping.')
            # Delete the file from XA
            XA.remove(xa)

    return XA, XI, Y


def make_partitions(XA, XI, Y):
    # Make folder
    os.makedirs(PARTITIONS_FOLDER_PATH, exist_ok=True)

    # Make 5-fold partitions
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_val_idx, test_idx) in enumerate(kf.split(XA)):
        XA_train_val, XA_test = [XA[idx] for idx in train_val_idx], [XA[idx] for idx in test_idx]
        XI_train_val, XI_test = [XI[idx] for idx in train_val_idx], [XI[idx] for idx in test_idx]
        Y_train_val, Y_test = [Y[idx] for idx in train_val_idx], [Y[idx] for idx in test_idx]

        # Split train into train and validation
        XA_train, XA_val, XI_train, XI_val, Y_train, Y_val = train_test_split(XA_train_val, XI_train_val, Y_train_val, test_size=0.2, random_state=42)

        # Save each partition in a file
        partition_folder = os.path.join(PARTITIONS_FOLDER_PATH, f'fold_{i+1}')
        os.makedirs(partition_folder, exist_ok=True)

        with open(os.path.join(partition_folder, 'train.txt'), 'w') as f:
            for xa, xi, y in zip(XA_train, XI_train, Y_train):
                f.write(f'{xa}\t{xi}\t{y}\n')

        with open(os.path.join(partition_folder, 'val.txt'), 'w') as f:
            for xa, xi, y in zip(XA_val, XI_val, Y_val):
                f.write(f'{xa}\t{xi}\t{y}\n')

        with open(os.path.join(partition_folder, 'test.txt'), 'w') as f:
            for xa, xi, y in zip(XA_test, XI_test, Y_test):
                f.write(f'{xa}\t{xi}\t{y}\n')


def load_partition(partition_path, use_distorted_images: bool = True):
    XA, XI, Y = [], [], []

    with open(partition_path, 'r') as f:
        for line in f.readlines():
            xa, xi, y = line.split('\t')
            # Check if distorted images are used
            if use_distorted_images:
                XA.append(xa)
            else:
                XA.append(xa.replace('_distorted', ''))
            XI.append(xi)
            Y.append(y)

    return XA, XI, Y


###################################################################### PREPROCESSING UTILS:


NUM_CHANNELS = 1
# TODO:
# - Check the image size!!
IMG_HEIGHT = 64
toTensor = transforms.ToTensor()


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
def get_image_from_file(path):
    x = Image.open(path).convert('L')                       # Convert to grayscale
    new_width = int(IMG_HEIGHT * x.size[0] / x.size[1])     # Get width preserving aspect ratio
    x = x.resize((new_width, IMG_HEIGHT))                   # Resize
    x = toTensor(x)                                         # Convert to tensor (normalizes to [0, 1])
    return x


def get_transcript_from_cleanKern(y_clean, w2i):
    y_coded = []

    # Clean up transcript
    y_clean = ''.join(y_clean)
    events = y_clean.split('\n')
    for i, event in enumerate(events):
        voices = event.split('\t')
        for j, voice in enumerate(voices):
            y_coded.append(voice)
            if j != len(voices) - 1:
                y_coded.append(COC_TOKEN)
        if i != len(events) - 1:
            y_coded.append(COR_TOKEN)

    y_coded = [SOT_TOKEN] + y_coded + [EOT_TOKEN]

    # Convert to indices
    y_coded = [w2i[token] for token in y_coded]

    # Convert to PyTorch tensor
    y_coded = torch.tensor(y_coded)

    return y_coded
    

###################################################################### PYTORCH DATASET UTILS:


class GrandStaffDataset(Dataset):
    def __init__(
            self,
            XA: List[str],
            XI: List[str],
            Y: List[str],
            w2i: Dict[str, int],
            i2w: Dict[int, str],
            kern_encoding: str = 'kern',
            keep_ligatures: bool = True) -> None:
        
        # Constants
        self.XA = XA
        self.XI = XI
        self.Y = Y
        self.w2i = w2i
        self.i2w = i2w

        # Kern Converter
        self.krn_encoder = krnConverter(kern_encoding, keep_ligatures)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        # Get spectrogram
        xa = get_spectrogram_from_file(self.XA[index])
        # Get image
        xi = get_image_from_file(self.XI[index])
        # Get transcript
        y = get_transcript_from_cleanKern(self.krn_encoder.convert(self.Y[index]), self.w2i)
        return xa, xi, y


###################################################################### PYTORCH DATALOADER UTILS:


def pad_batch_images(x):
    max_width = max(x, key=lambda sample: sample.shape[2]).shape[2]
    x = torch.stack([F.pad(i, pad=(0, max_width - i.shape[2])) for i in x], dim=0)
    return x


def pad_batch_transcripts(x):
    max_length = max(x, key=lambda sample: sample.shape[0]).shape[0]
    x = torch.stack([F.pad(i, pad=(0, max_length - i.shape[0])) for i in x], dim=0)
    x = x.long()
    return x


def batch_preparation(batch):
    xa, xi, y = zip(*batch)
    # Zero-pad spectrograms/images to maximum batch spectrogram/image width
    xa = pad_batch_images(xa)
    xi = pad_batch_images(xi)
    # Decoder input: <sot> symbols
    dec_in = pad_batch_transcripts(y[:-1])
    # Decoder output: symbols <eot>
    dec_out = pad_batch_transcripts(y[1:])
    return xa, xi, dec_in, dec_out


if __name__ == '__main__':
    import argparse

    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid, save_image

    from encoding import ENCODING_OPTIONS


    CHECK_DIR = "check"
    if not os.path.isdir(CHECK_DIR):
        os.mkdir(CHECK_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_distorted_images', action='store_true', help='Use distorted images')
    parser.add_argument('--fold_id', type=int, default=1, choices=[1, 2, 3, 4, 5], help='Fold id')
    parser.add_argument('--kern_encoding', type=str, default='kern', choices=ENCODING_OPTIONS, help='Kern encoding')
    parser.add_argument('--keep_ligatures', action='store_true', help='Keep ligatures')
    args = parser.parse_args()

    partitions, w2i, i2w = load_fold_data(
        use_distorted_images=args.use_distorted_images,
        fold_id=args.fold_id,
        kern_encoding=args.kern_encoding,
        keep_ligatures=args.keep_ligatures)
    
    train_ds = GrandStaffDataset(
        XA=partitions['train'][0],
        XI=partitions['train'][1],
        Y=partitions['train'][2],
        w2i=w2i,
        i2w=i2w,
        kern_encoding=args.kern_encoding,
        keep_ligatures=args.keep_ligatures)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=batch_preparation)

    print(f'Train dataset size: {len(train_ds)}')
    for xa, xi, dec_in, dec_out in train_loader:
        print('Types:')
        print('\txa:', xa.dtype)
        print('\txi:', xi.dtype)
        print('\tdec_in:', dec_in.dtype)
        print('\tdec_out:', dec_out.dtype)
        print('Shapes:')
        print('\txa:', xa.shape)
        print('\txi:', xi.shape)
        print('\tdec_in:', dec_in.shape)
        print('\tdec_out:', dec_out.shape)

        # Save batch spectrogram/images
        save_image(make_grid(xa, nrow=4), f'{CHECK_DIR}/xa_train_batch.jpg')
        save_image(make_grid(xi, nrow=4), f'{CHECK_DIR}/xi_train_batch.jpg')

        # See first sample
        print(f'Shape with padding: {dec_in[0].shape}')
        print('Decoder input:', [train_ds.i2w[i.item()] for i in dec_in[0]])
        print('Decoder output:', [train_ds.i2w[i.item()] for i in dec_out[0]])
        save_image(xi[0], f'{CHECK_DIR}/xi0_train_batch.jpg')

        break
