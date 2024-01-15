import os
import json
import math

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

from encoding import krnParser
from prepare_dataset import GRANDSTAFF_PATH
from transformer.encoder import HEIGHT_REDUCTION, WIDTH_REDUCTION
from preprocessing import (
    preprocess_audio,
    preprocess_image,
    ar_batch_preparation_multimodal,
    ar_batch_preparation_image,
    ar_batch_preparation_audio,
)


SOS_TOKEN = "<sos>"  # Start-of-sequence token
EOS_TOKEN = "<eos>"  # End-of-sequence token


DATASETS = [
    "grandstaff",
    "beethoven",
    "chopin",
    "hummel",
    "joplin",
    "mozart",
    "scarlatti-d",
]


class ARDataModule(LightningDataModule):
    def __init__(
        self,
        ds_name: str,
        krn_encoding: str = "bekern",
        input_modality: str = "both",  # "audio" or "image" or "both"
        use_distorted_images: bool = False,
        img_height: int = None,  # If None, the original image height is used
        batch_size: int = 16,
        num_workers: int = 20,
    ):
        super(ARDataModule, self).__init__()
        self.ds_name = ds_name
        self.krn_encoding = krn_encoding
        self.input_modality = input_modality
        self.use_distorted_images = use_distorted_images
        self.img_height = img_height
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = (
            ar_batch_preparation_multimodal
            if input_modality == "both"
            else (
                ar_batch_preparation_image
                if input_modality == "image"
                else ar_batch_preparation_audio
            )
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = ARDataset(
                ds_name=self.ds_name,
                partition_type="train",
                krn_encoding=self.krn_encoding,
                input_modality=self.input_modality,
                use_distorted_images=self.use_distorted_images,
                img_height=self.img_height,
            )
            self.val_ds = ARDataset(
                ds_name=self.ds_name,
                partition_type="val",
                krn_encoding=self.krn_encoding,
                input_modality=self.input_modality,
                use_distorted_images=self.use_distorted_images,
                img_height=self.img_height,
            )

        if stage == "test" or stage == "predict":
            self.test_ds = ARDataset(
                ds_name=self.ds_name,
                partition_type="test",
                krn_encoding=self.krn_encoding,
                input_modality=self.input_modality,
                use_distorted_images=self.use_distorted_images,
                img_height=self.img_height,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )  # prefetch_factor=2

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def predict_dataloader(self):
        print("Using test_dataloader for predictions.")
        return self.test_dataloader(self)

    def get_w2i_and_i2w(self):
        try:
            return self.train_ds.w2i, self.train_ds.i2w
        except AttributeError:
            return self.test_ds.w2i, self.test_ds.i2w

    def get_max_seq_len(self):
        try:
            return self.train_ds.max_seq_len
        except AttributeError:
            return self.test_ds.max_seq_len

    def get_max_image_height_and_width(self):
        try:
            return self.train_ds.max_image_height, self.train_ds.max_image_width
        except AttributeError:
            return self.test_ds.max_image_height, self.test_ds.max_image_width

    def get_max_audio_height_and_width(self):
        try:
            return self.train_ds.max_audio_height, self.train_ds.max_audio_width
        except AttributeError:
            return self.test_ds.max_audio_height, self.test_ds.max_audio_width


####################################################################################################


class ARDataset(Dataset):
    def __init__(
        self,
        ds_name: str,
        partition_type: str,
        krn_encoding: str = "bekern",
        input_modality: str = "both",  # "audio" or "image" or "both"
        use_distorted_images: bool = False,
        img_height: int = None,  # If None, the original image height is used
    ):
        self.ds_name = ds_name.lower()
        self.partition_type = partition_type
        self.input_modality = input_modality.lower()
        self.use_distorted_images = (
            use_distorted_images  # Only used if input_modality == "image" or "both"
        )
        self.img_height = img_height  # Only used if input_modality == "image" or "both"
        self.init(krn_encoding=krn_encoding, vocab_name="ar_w2i")

    # ---------------------------------------------------------------------------- INITIALIZATION

    def init(self, krn_encoding: str = "bekern", vocab_name: str = "w2i"):
        # Initialize krn parser
        self.krn_parser = krnParser(encoding=krn_encoding)

        # Check dataset name
        assert self.ds_name in DATASETS, f"Invalid dataset name: {self.ds_name}"

        # Check partition type
        assert self.partition_type in [
            "train",
            "val",
            "test",
        ], f"Invalid partition type: {self.partition_type}"

        # Set folder paths and input extensions
        ds_folder_path = os.path.join(GRANDSTAFF_PATH, self.ds_name)
        self.audio_folder_path = os.path.join(ds_folder_path, "wav")
        self.image_folder_path = (
            os.path.join(ds_folder_path, "img_distorted")
            if self.use_distorted_images
            else os.path.join(ds_folder_path, "img")
        )
        self.img_extension = "_distorted.jpg" if self.use_distorted_images else ".jpg"
        self.transcript_folder_path = (
            os.path.join(ds_folder_path, "bekrn")
            if self.krn_parser.encoding == "bekern"
            else os.path.join(ds_folder_path, "krn")
        )
        self.transcript_extension = (
            ".bekrn" if self.krn_parser.encoding == "bekern" else ".krn"
        )

        # Get audios or images or both and transcripts files
        # X = (list of images_paths, list of audios_paths); Y is a list of transcripts_paths
        # images_path or audios_path is None if input_modality != "image" or "audio"
        assert self.input_modality in [
            "image",
            "audio",
            "both",
        ], f"Invalid input_modality: {self.input_modality}"
        (
            self.X,
            self.Y,
        ) = self.get_inputs_and_transcripts_files()

        # Check and retrieve vocabulary
        vocab_folder = os.path.join(GRANDSTAFF_PATH, "vocabs")
        os.makedirs(vocab_folder, exist_ok=True)
        vocab_name = self.ds_name + f"_{vocab_name}_{krn_encoding}.json"
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary()

        # Set max_seq_len, max_image_len and max_audio_len
        self.set_max_lens()

    def get_inputs_and_transcripts_files(self):
        images = []
        audios = []
        transcripts = []
        partition_file = os.path.join(
            GRANDSTAFF_PATH, "partitions", self.ds_name, self.partition_type + ".txt"
        )
        with open(partition_file, "r") as file:
            for s in file.read().splitlines():
                s = s.strip()
                images.append(
                    os.path.join(self.image_folder_path, s + self.img_extension)
                )
                audios.append(os.path.join(self.audio_folder_path, s + ".wav"))
                transcripts.append(
                    os.path.join(
                        self.transcript_folder_path, s + self.transcript_extension
                    )
                )
        if self.input_modality == "image":
            return (images, None), transcripts
        elif self.input_modality == "audio":
            return (None, audios), transcripts
        else:
            # self.input_modality == "both"
            return (images, audios), transcripts

    def check_and_retrieve_vocabulary(self):
        w2i = {}
        i2w = {}

        if os.path.isfile(self.w2i_path):
            with open(self.w2i_path, "r") as file:
                w2i = json.load(file)
            i2w = {v: k for k, v in w2i.items()}
        else:
            w2i, i2w = self.make_vocabulary()
            with open(self.w2i_path, "w") as file:
                json.dump(w2i, file)

        return w2i, i2w

    def make_vocabulary(self):
        vocab = []
        for partition_type in ["train", "val", "test"]:
            partition_file = os.path.join(
                GRANDSTAFF_PATH,
                "partitions",
                self.ds_name,
                partition_type + ".txt",
            )
            with open(partition_file, "r") as file:
                for s in file.read().splitlines():
                    s = s.strip()
                    transcript = self.krn_parser.encode(
                        file_path=os.path.join(
                            self.transcript_folder_path, s + self.transcript_extension
                        )
                    )
                    vocab.extend(transcript)
        vocab = [SOS_TOKEN, EOS_TOKEN] + vocab
        vocab = sorted(set(vocab))

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w

    def set_max_lens(self):
        # Set the maximum lengths for the whole GRANDSTAFF collection:
        # 1) Get the maximum transcript length
        # 2) Get the maximum image size
        # 3) Get the maximum audio size
        max_seq_len = 0
        max_image_height, max_image_width = 0, 0
        max_audio_height, max_audio_width = 0, 0
        for foldername, subfolders, filenames in os.walk(GRANDSTAFF_PATH):
            for filename in filenames:
                if filename.startswith("."):
                    continue

                if filename.endswith(self.transcript_extension):
                    transcript = self.krn_parser.encode(
                        file_path=os.path.join(foldername, filename)
                    )
                    max_seq_len = max(
                        max_seq_len, len(transcript) + 1
                    )  # +1 for EOS token
                elif filename.endswith(self.img_extension):
                    image = preprocess_image(
                        path=os.path.join(foldername, filename),
                    )
                    max_image_height = max(max_image_height, image.shape[1])
                    max_image_width = max(max_image_width, image.shape[2])
                elif filename.endswith(".wav"):
                    audio = preprocess_audio(path=os.path.join(foldername, filename))
                    max_audio_height = max(max_audio_height, audio.shape[1])
                    max_audio_width = max(max_audio_width, audio.shape[2])
                else:
                    continue

        self.max_seq_len = max_seq_len
        self.max_image_height = max_image_height
        self.max_image_width = max_image_width
        self.max_audio_height = max_audio_height
        self.max_audio_width = max_audio_width

    # ---------------------------------------------------------------------------- GETTERS

    def __len__(self):
        return len(self.Y)

    def __getitemimage__(self, idx):
        x = preprocess_image(path=self.X[0][idx], img_height=self.img_height)
        y = self.preprocess_transcript(path=self.Y[idx])
        if self.partition_type == "train":
            return x, self.get_number_of_frames(x), y
        return x, y

    def __getitemaudio__(self, idx):
        x = preprocess_audio(path=self.X[1][idx])
        y = self.preprocess_transcript(path=self.Y[idx])
        if self.partition_type == "train":
            return x, self.get_number_of_frames(x), y
        return x, y

    def __getitemboth__(self, idx):
        xi = preprocess_image(path=self.X[0][idx])
        xa = preprocess_audio(path=self.X[1][idx])
        y = self.preprocess_transcript(path=self.Y[idx])
        if self.partition_type == "train":
            return (
                xi,
                self.get_number_of_frames(xi),
                xa,
                self.get_number_of_frames(xa),
                y,
            )
        return xi, xa, y

    def __getitem__(self, idx):
        return getattr(self, "__getitem" + self.input_modality + "__")(idx)

    def preprocess_transcript(self, path: str):
        y = self.krn_parser.encode(src_file=path)
        y = [SOS_TOKEN] + y + [EOS_TOKEN]
        y = [self.w2i[w] for w in y]
        return torch.tensor(y, dtype=torch.int64)

    def get_number_of_frames(self, x):
        # x is the output of preprocess_image or preprocess_audio
        # x.shape = [1, height, width] or [1, freq_bins, time_frames]
        return math.ceil(x.shape[1] / HEIGHT_REDUCTION) * math.ceil(
            x.shape[2] / WIDTH_REDUCTION
        )
