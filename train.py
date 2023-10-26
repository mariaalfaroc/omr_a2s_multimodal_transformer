import os
import gin
import fire
import torch

from loguru import logger
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from data.grandstaff import NUM_CHANNELS, batch_preparation_audio, batch_preparation_image, batch_preparation_multimodal, load_gs_datasets
from model.dan_transformer import Poliphony_DAN, get_model

BATCH_FUNCTIONS = {
    "DAN_audio": batch_preparation_audio,
    "DAN_image": batch_preparation_image,
    "DAN_multimodal": batch_preparation_multimodal
}

torch.set_float32_matmul_precision("high")

@gin.configurable
def main(data_path, checkpoint_path=None, corpus_name=None, model_name=None, keep_vocabulary=False, metric_to_watch=None, max_epochs=10000):
    logger.info("-----------------------")
    logger.info(f"Training with {data_path}")
    logger.info(f"Training with the {model_name} model")
    logger.info("-----------------------")

    # Create output directory
    out_dir = f"out/{model_name}"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/hyp", exist_ok=True)
    os.makedirs(f"{out_dir}/gt", exist_ok=True)

    # Load data
    logger.info("Loading data...")
    vocab_path = f"vocab/{corpus_name}"
    if not keep_vocabulary:
        # remove vocab folder if exists
        if os.path.exists(vocab_path):
            os.system(f"rm -rf {vocab_path}")
    train_dataset, val_dataset, test_dataset = load_gs_datasets(path=data_path, kern_encoding='bekern', use_distorted_images=False, vocab_path=vocab_path)
    print(f'Train: {len(train_dataset)}; Test: {len(test_dataset)}; Val: {len(val_dataset)}')

    # Get dictionaries
    w2i, i2w = train_dataset.get_dictionaries()

    # Create dataloaders (16 in boo, recommended by PyTorch)
    assert(model_name in BATCH_FUNCTIONS.keys()), f"Model name {model_name} must be one in {BATCH_FUNCTIONS.keys()}"
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=16, collate_fn=BATCH_FUNCTIONS[model_name])
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=16, collate_fn=BATCH_FUNCTIONS[model_name])
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=16, collate_fn=BATCH_FUNCTIONS[model_name])

    # Get max values
    # This code calculates the maximum height, width, and sequence length of the images in three different datasets: train_dataset, val_dataset, and test_dataset. It then assigns the maximum values to the variables maxheight, maxwidth, and maxlen, respectively.
    logger.info("Getting max length of sequences...")
    train_hw = train_dataset.get_max_audio_hw()
    val_hw = val_dataset.get_max_audio_hw()
    test_hw = test_dataset.get_max_audio_hw()
    maxheight = max([train_hw[0], val_hw[0], test_hw[0]])
    maxwidth = max([train_hw[1], val_hw[1], test_hw[1]])
    maxlen = max([train_dataset.get_max_seqlen(), val_dataset.get_max_seqlen(), test_dataset.get_max_seqlen()])
    vocab_size = train_dataset.get_vocab_size()

    # Get model
    model = get_model(in_channels=NUM_CHANNELS, d_model=256, dim_ff=256,
                        max_height=maxheight, max_width=maxwidth, 
                        max_len=maxlen, 
                        out_categories=vocab_size, w2i=w2i, i2w=i2w, out_dir=out_dir)

    # Initializes a logger object from the WandbLogger class.
    # The logger is used to log the training progress of a machine learning model.
    # The logger is created with the following parameters:
    # - project: the name of the project in the Weights and Biases platform.
    # - group: the name of the group to which the logger belongs.
    # - name: the name of the logger.
    # - log_model: a boolean value that indicates whether to log the model or not. In this case, it is set to False.
    wandb_logger = WandbLogger(project="AMT", group=corpus_name, name=f"{model_name}", log_model=False)

    # Stops the training process if the validation loss does not improve for five consecutive epochs.
    early_stopping = EarlyStopping(monitor=metric_to_watch, min_delta=0.01, patience=5, mode="min", verbose=True)
    
    # Saves the model with the lowest validation loss.
    checkpointer = ModelCheckpoint(dirpath=f"weights/{corpus_name}/", filename=f"{model_name}", 
                                   monitor=metric_to_watch, mode="min",
                                   save_top_k=1, verbose=True)

    # Creates a Trainer object for a maximum of 10,000 epochs.
    trainer = Trainer(max_epochs=max_epochs, check_val_every_n_epoch=5, logger=wandb_logger, callbacks=[checkpointer, early_stopping])

    # Loads the weights of the model from the checkpoint_path if provided
    if checkpoint_path is not None:
        logger.info(f"Loading weights from {checkpoint_path}")
        model = Poliphony_DAN.load_from_checkpoint(checkpoint_path)
    
    # Trains the model
    trainer.fit(model, train_dataloader, val_dataloader)

    # Loads the best model from the training process
    model = Poliphony_DAN.load_from_checkpoint(checkpointer.best_model_path)

    # Tests the model
    trainer.test(model, test_dataloader)

def launch(config, checkpoint_path=None):
    gin.parse_config_file(config)
    main(checkpoint_path=checkpoint_path)

if __name__ == "__main__":
    fire.Fire(launch)
