import gc
import os
from typing import Optional

import fire
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from data.ar_dataset import ARDataModule
from data.preprocessing import MEMORY
from transformer.model import MultimodalTransformer, Transformer
from utils.seed import seed_everything

seed_everything(42, deterministic=False, benchmark=False)

# Set WANDB_API_KEY
with open("wandb_api_key.txt", "r") as f:
    os.environ["WANDB_API_KEY"] = f.read().strip()


def train(
    ds_name,
    krn_encoding: str = "bekern",
    input_modality: str = "audio",  # "audio" or "image" or "both"
    use_distorted_images: bool = False,  # Only used if input_modality == "image" or "both"
    img_height: Optional[
        int
    ] = None,  # If None, the original image height is used (only used if input_modality == "image" or "both")
    attn_window: int = -1,  # Number of past tokens to attends to; -1 == no limitation (attends to all past tokens)
    mixer_type: Optional[
        str
    ] = None,  # Only used if input_modality == "both" -> to choose the mixer type of the MultimodalTransformer (can be "concat", "attn_img", "attn_audio", or "attn_both")
    epochs: int = 1000,
    patience: int = 20,
    batch_size: int = 16,
    checkpoint_path: str = "",  # If not empty, the model will be loaded from this checkpoint
):
    gc.collect()
    torch.cuda.empty_cache()

    # Experiment info
    print("TRAIN EXPERIMENT")
    print(f"\tDataset: {ds_name}")
    print(f"\tKern encoding: {krn_encoding}")
    print(f"\tInput modality: {input_modality}")
    print(
        f"\tUse distorted images: {use_distorted_images} (used if input_modality in ['image', 'both'])"
    )
    print(f"\tImage height: {img_height} (used if input_modality in ['image', 'both'])")
    print(f"\tAttention window: {attn_window}")
    print(f"\tMixer type: {mixer_type} (used if input_modality == 'both')")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tBatch size: {batch_size}")
    print(f"\tCheckpoint path (to resume training): {checkpoint_path}")

    # Data module
    datamodule = ARDataModule(
        ds_name=ds_name,
        krn_encoding=krn_encoding,
        input_modality=input_modality,
        use_distorted_images=use_distorted_images,
        img_height=img_height,
        batch_size=batch_size,
    )
    datamodule.setup(stage="fit")
    w2i, i2w = datamodule.get_w2i_and_i2w()

    # Model
    model_class = MultimodalTransformer if input_modality == "both" else Transformer
    if os.path.exists(checkpoint_path):
        try:
            print("Checkpoint found. Loading model from checkpoint...")
            model = model_class.load_from_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Training will start from scratch.")
            checkpoint_path = ""
    else:
        print("No checkpoint found. Training will start from scratch.")

    if not checkpoint_path:
        if input_modality == "both":
            (max_h_img, max_w_img), (max_h_audio, max_w_audio) = (
                datamodule.get_max_input_size()
            )
            model = model_class(
                max_img_height=max_h_img,
                max_img_width=max_w_img,
                max_audio_height=max_h_audio,
                max_audio_width=max_w_audio,
                max_seq_len=datamodule.get_max_seq_len(),
                w2i=w2i,
                i2w=i2w,
                mixer_type=mixer_type,
                attn_window=attn_window,
                teacher_forcing_prob=0.2,
                teacher_forcing_modality_prob=0.2,
            )
        else:
            max_h, max_w = datamodule.get_max_input_size()
            model = model_class(
                max_input_height=max_h,
                max_input_width=max_w,
                max_seq_len=datamodule.get_max_seq_len(),
                w2i=w2i,
                i2w=i2w,
                attn_window=attn_window,
                teacher_forcing_prob=0.2,
            )

    # Model name
    model_name = input_modality
    model_name += (
        "_distorted" if input_modality == "image" and use_distorted_images else ""
    )
    model_name += (
        f"_height{img_height}"
        if input_modality == "image" and img_height is not None
        else ""
    )
    model_name += (
        f"_{mixer_type}" if mixer_type is not None and input_modality == "both" else ""
    )
    model_name += f"_{krn_encoding}"

    # Train, validate and test
    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{ds_name}",
            filename=model_name,
            monitor="val_sym-er",
            verbose=True,
            save_last=False,
            save_top_k=1,
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=False,
            every_n_epochs=5,
            save_on_train_epoch_end=False,
        ),
        EarlyStopping(
            monitor="val_sym-er",
            min_delta=0.01,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            check_on_train_epoch_end=False,
        ),
    ]
    trainer = Trainer(
        logger=WandbLogger(
            project="OMR-A2S-Poly-Multimodal",
            group=model_name,
            name=f"Train-{ds_name}_Test-{ds_name}",
            log_model=False,
            entity="grfia",
        ),
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=5,
        deterministic=False,
        benchmark=False,
        precision="16-mixed",  # Mixed precision training
    )
    trainer.fit(model, datamodule=datamodule)
    model = model_class.load_from_checkpoint(callbacks[0].best_model_path)
    model.freeze()
    trainer.test(model, datamodule=datamodule)

    # Free cache memory
    MEMORY.clear()


if __name__ == "__main__":
    fire.Fire(train)
