import json
import math
import os
from typing import Dict, Optional, Tuple, Union

import torch
from datasets import load_dataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from data.encoding import krnParser, ENCODING_OPTIONS
from data.prepare_dataset import GRANDSTAFF_PATH
from data.preprocessing import (
    ar_batch_preparation_audio,
    ar_batch_preparation_image,
    ar_batch_preparation_multimodal,
    preprocess_audio,
    preprocess_image,
)
from transformer.encoder import HEIGHT_REDUCTION, WIDTH_REDUCTION

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
SPLITS = ["train", "val", "test"]
MODALITIES = ["audio", "image", "both"]


class ARDataModule(LightningDataModule):
    """
    Auto-regressive data module for the GRANDSTAFF collection.

    Args:
        ds_name (str): Dataset name. It must be one of the following: "grandstaff", "beethoven", "chopin", "hummel", "joplin", "mozart", "scarlatti-d".
        krn_encoding (str, optional): Encoding for the krn files. It must be one of the following: "bekern", "krn". Defaults to "bekern".
        input_modality (str, optional): Input modality. It must be one of the following: "audio", "image", "both". Defaults to "both".
        use_distorted_images (bool, optional): If True, the distorted images are used. Only used if input_modality == "image" or "both". Defaults to False.
        img_height (Optional[int], optional): Image height. If None, the original image height is used. Only used if input_modality == "image" or "both". Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 16.
        num_workers (int, optional): Number of workers. Defaults to 20.
    """

    def __init__(
        self,
        ds_name: str,
        krn_encoding: str = "bekern",
        input_modality: str = "both",  # "audio" or "image" or "both"
        use_distorted_images: bool = False,
        img_height: Optional[int] = None,  # If None, the original image height is used
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
            else (ar_batch_preparation_image if input_modality == "image" else ar_batch_preparation_audio)
        )

        # Datasets
        # To prevent executing setup() twice
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str):
        if stage == "fit":
            if not self.train_ds:
                self.train_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="train",
                    krn_encoding=self.krn_encoding,
                    input_modality=self.input_modality,
                    use_distorted_images=self.use_distorted_images,
                    img_height=self.img_height,
                )
            if not self.val_ds:
                self.val_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="val",
                    krn_encoding=self.krn_encoding,
                    input_modality=self.input_modality,
                    use_distorted_images=self.use_distorted_images,
                    img_height=self.img_height,
                )

        if stage == "test" or stage == "predict":
            if not self.test_ds:
                self.test_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="test",
                    krn_encoding=self.krn_encoding,
                    input_modality=self.input_modality,
                    use_distorted_images=self.use_distorted_images,
                    img_height=self.img_height,
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )  # prefetch_factor=2

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def predict_dataloader(self) -> DataLoader:
        print("Using test_dataloader for predictions.")
        return self.test_dataloader()

    def get_w2i_and_i2w(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        try:
            return self.train_ds.w2i, self.train_ds.i2w
        except AttributeError:
            return self.test_ds.w2i, self.test_ds.i2w

    def get_max_seq_len(self) -> int:
        try:
            return self.train_ds.max_seq_len
        except AttributeError:
            return self.test_ds.max_seq_len

    def get_max_image_height_and_width(self) -> Tuple[int, int]:
        try:
            return self.train_ds.max_image_height, self.train_ds.max_image_width
        except AttributeError:
            return self.test_ds.max_image_height, self.test_ds.max_image_width

    def get_max_audio_height_and_width(self) -> Tuple[int, int]:
        try:
            return self.train_ds.max_audio_height, self.train_ds.max_audio_width
        except AttributeError:
            return self.test_ds.max_audio_height, self.test_ds.max_audio_width

    def get_max_input_size(
        self,
    ) -> Union[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Returns the maximum height and width for the input modality:
            - If input_modality == "image" or "audio", it returns a tuple of two integers.
            - If input_modality == "both", it returns a tuple of two tuples of two integers. First tuple for images and second tuple for audios.
        """
        if self.input_modality == "image":
            return self.get_max_image_height_and_width()
        elif self.input_modality == "audio":
            return self.get_max_audio_height_and_width()
        elif self.input_modality == "both":
            return (
                self.get_max_image_height_and_width(),
                self.get_max_audio_height_and_width(),
            )


####################################################################################################


class ARDataset(Dataset):
    """
    Auto-regressive dataset for the GRANDSTAFF collection.

    Args:
        ds_name (str): Dataset name. It must be one of the following: "grandstaff", "beethoven", "chopin", "hummel", "joplin", "mozart", "scarlatti-d".
        partition_type (str): Partition type. It must be one of the following: "train", "val", "test".
        krn_encoding (str, optional): Encoding for the krn files. It must be one of the following: "bekern", "krn". Defaults to "bekern".
        input_modality (str, optional): Input modality. It must be one of the following: "audio", "image", "both". Defaults to "both".
        use_distorted_images (bool, optional): If True, the distorted images are used. Only used if input_modality == "image" or "both". Defaults to False.
        img_height (Optional[int], optional): Image height. If None, the original image height is used. Only used if input_modality == "image" or "both". Defaults to None.
    """

    def __init__(
        self,
        ds_name: str,
        partition_type: str,
        krn_encoding: str = "bekern",
        input_modality: str = "both",  # "audio" or "image" or "both"
        use_distorted_images: bool = False,
        img_height: Optional[int] = None,  # If None, the original image height is used
    ):
        self.ds_name = ds_name.lower()
        self.partition_type = partition_type
        self.input_modality = input_modality.lower()
        self.use_distorted_images = use_distorted_images  # Only used if input_modality == "image" or "both"
        self.img_height = img_height  # Only used if input_modality == "image" or "both"
        self.init(krn_encoding=krn_encoding, vocab_name="ar_w2i")

    # ---------------------------------------------------------------------------- INITIALIZATION

    def init(self, krn_encoding: str = "bekern", vocab_name: str = "w2i"):
        # Initialize krn parser
        self.krn_parser = krnParser(encoding=krn_encoding)

        # Check dataset name
        assert self.ds_name in DATASETS, f"Invalid dataset name: {self.ds_name}"

        # Check partition type
        assert self.partition_type in SPLITS, f"Invalid partition type: {self.partition_type}"

        # Load dataset
        assert self.input_modality in MODALITIES, f"Invalid input_modality: {self.input_modality}"
        self.ds = load_dataset(f"PRAIG/{self.ds_name}-grandstaff-multimodal", split=self.partition_type)
        # Rename correct encoding column to transcript
        self.ds = self.ds.rename_column(self.krn_parser.encoding, "transcript")
        # Remove all other encoding options
        remove_columns = [e for e in ENCODING_OPTIONS if e != self.krn_parser.encoding]

        # Get the correct dataset features
        if self.input_modality == "audio":
            print(f"Using audio modality for {self.ds_name} dataset.")
            # Remove also all image columns
            remove_columns += ["image", "image_distorted"]
            print(f"Removing columns: {remove_columns}")
            self.ds = self.ds.remove_columns(remove_columns)
            print(f"Columns: {self.ds.column_names}")

        elif self.input_modality in ["image", "both"]:
            if self.input_modality == "image":
                print(f"Using image modality for {self.ds_name} dataset.")
                # Remove also all audio columns
                remove_columns += ["audio"]
            elif self.input_modality == "both":
                print(f"Using both audio and image modalities for {self.ds_name} dataset.")
            else:
                raise ValueError(f"Invalid input_modality: {self.input_modality}. Must be 'image' or 'both'.")
            
            # Check if distorted images are used
            image_key = None
            if self.use_distorted_images:
                print("Using distorted images.")
                remove_columns += ["image"]
                image_key = "image_distorted"
            else:
                remove_columns += ["image_distorted"]
                image_key = "image"
            print(f"Removing columns: {remove_columns}")
            self.ds = self.ds.remove_columns(remove_columns)
            # Image column should always be named "image"
            self.ds = self.ds.rename_column(image_key, "image")
            print(f"Columns: {self.ds.column_names}")

        else:
            raise ValueError(f"Invalid input_modality: {self.input_modality}. Must be one of {MODALITIES}.")

        # Check and retrieve vocabulary
        vocab_folder = os.path.join(GRANDSTAFF_PATH, "vocabs")
        os.makedirs(vocab_folder, exist_ok=True)
        vocab_name = f"{vocab_name}_{krn_encoding}.json"
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary()

        # Check and retrive max lengths
        max_lens_folder = os.path.join(GRANDSTAFF_PATH, "max_lens")
        os.makedirs(max_lens_folder, exist_ok=True)
        max_lens_name = "ImgDist_" if self.use_distorted_images else ""
        max_lens_name += vocab_name
        self.max_lens_path = os.path.join(max_lens_folder, max_lens_name)
        max_lens = self.check_and_retrieve_max_lens()
        self.max_seq_len = max_lens["max_seq_len"]
        self.max_image_height = max_lens["max_image_height"]
        self.max_image_width = max_lens["max_image_width"]
        self.max_audio_height = max_lens["max_audio_height"]
        self.max_audio_width = max_lens["max_audio_width"]

    def check_and_retrieve_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
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

    def make_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Use the same vocabulary for the whole GRANDSTAFF collection
        full_ds = load_dataset("PRAIG/grandstaff-grandstaff-multimodal")

        vocab = []
        for partition_type in SPLITS:
            for text in full_ds[partition_type][self.krn_parser.encoding]:
                transcript = self.krn_parser.encode(text=text)
                vocab.extend(transcript)
        vocab = sorted(set(vocab))

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

    def check_and_retrieve_max_lens(self) -> Dict[str, int]:
        max_lens = {}

        if os.path.isfile(self.max_lens_path):
            with open(self.max_lens_path, "r") as file:
                max_lens = json.load(file)
        else:
            max_lens = self.make_max_lens()
            with open(self.max_lens_path, "w") as file:
                json.dump(max_lens, file)

        return max_lens

    def make_max_lens(self):
        # Set the maximum lengths for the whole GRANDSTAFF collection:
        # 1) Get the maximum transcript length
        # 2) Get the maximum image size
        # 3) Get the maximum audio size
        max_seq_len = 0
        max_image_height, max_image_width = 0, 0
        max_audio_height, max_audio_width = 0, 0

        full_ds = load_dataset("PRAIG/grandstaff-grandstaff-multimodal")
        for partition_type in SPLITS:
            for sample in full_ds[partition_type]:
                # Max transcript length
                transcript = sample[self.krn_parser.encoding]
                text = self.krn_parser.encode(text=transcript)
                max_seq_len = max(max_seq_len, len(text) + 1)  # +1 for EOS token

                # Max audio size
                raw_audio = sample["audio"]
                audio = preprocess_audio(
                    raw_audio=raw_audio["array"],
                    sr=raw_audio["sampling_rate"],
                    dtype=torch.float32,
                )
                max_audio_height = max(max_audio_height, audio.shape[1])
                max_audio_width = max(max_audio_width, audio.shape[2])

                # Max image size
                raw_image = sample["image_distorted"] if self.use_distorted_images else sample["image"]
                image = preprocess_image(
                    raw_image=raw_image,
                    img_height=self.img_height,
                    dtype=torch.float32,
                )
                max_image_height = max(max_image_height, image.shape[1])
                max_image_width = max(max_image_width, image.shape[2])

        return {
            "max_seq_len": max_seq_len,
            "max_image_height": max_image_height,
            "max_image_width": max_image_width,
            "max_audio_height": max_audio_height,
            "max_audio_width": max_audio_width,
        }

    # ---------------------------------------------------------------------------- GETTERS

    def __len__(self) -> int:
        return len(self.ds)

    def __getitemimage__(self, idx: int):
        sample = self.ds[idx]
        x = preprocess_image(raw_image=sample["image"], img_height=self.img_height, dtype=torch.float32)
        y = self.preprocess_transcript(text=sample["transcript"])
        if self.partition_type == "train":
            return x, self.get_number_of_frames(x), y
        return x, y

    def __getitemaudio__(self, idx: int):
        sample = self.ds[idx]
        x = preprocess_audio(
            raw_audio=sample["audio"]["array"], sr=sample["audio"]["sampling_rate"], dtype=torch.float32
        )
        y = self.preprocess_transcript(text=sample["transcript"])
        if self.partition_type == "train":
            return x, self.get_number_of_frames(x), y
        return x, y

    def __getitemboth__(self, idx: int):
        sample = self.ds[idx]
        xi = preprocess_image(raw_image=sample["image"], img_height=self.img_height, dtype=torch.float32)
        xa = preprocess_audio(
            raw_audio=sample["audio"]["array"], sr=sample["audio"]["sampling_rate"], dtype=torch.float32
        )
        y = self.preprocess_transcript(text=sample["transcript"])
        if self.partition_type == "train":
            return (
                xi,
                self.get_number_of_frames(xi),
                xa,
                self.get_number_of_frames(xa),
                y,
            )
        return xi, xa, y

    def __getitem__(self, idx: int):
        return getattr(self, "__getitem" + self.input_modality + "__")(idx)

    def preprocess_transcript(self, text: str) -> torch.Tensor:
        y = self.krn_parser.encode(text=text)
        y = [SOS_TOKEN] + y + [EOS_TOKEN]
        y = [self.w2i[w] for w in y]
        return torch.tensor(y, dtype=torch.int64)

    def get_number_of_frames(self, x: torch.Tensor) -> int:
        # x is the output of preprocess_image or preprocess_audio
        # x.shape = [1, height, width] or [1, freq_bins, time_frames]
        return math.ceil(x.shape[1] / HEIGHT_REDUCTION) * math.ceil(x.shape[2] / WIDTH_REDUCTION)
