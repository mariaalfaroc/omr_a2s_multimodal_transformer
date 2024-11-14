import sys
sys.path.append("./")

import gc
import os
import random

import fire
import torch
import swalign
from rich.progress import track
from lightning.pytorch.loggers.wandb import WandbLogger

from utils.metrics import compute_metrics
from transformer.model import Transformer
from data.ar_dataset import ARDataModule
from multimodal.smith_waterman.smith_waterman import (
    swalign_preprocess,
    undo_swalign_preprocess,
    dump,
    preprocess_prob,
    get_alignment,
)
from utils.seed import seed_everything

seed_everything(42, benchmark=False)

# Set WANDB_API_KEY
with open("wandb_api_key.txt", "r") as f:
    os.environ["WANDB_API_KEY"] = f.read().strip()


def test(
    ds_name: str,
    krn_encoding: str = "bekern",
    use_distorted_images: bool = False,
    img_height: int = None,  # If None, the original image height is used
    image_checkpoint_path: str = "",
    audio_checkpoint_path: str = "",
    match: int = 2,
    mismatch: int = -1,
    gap_penalty: int = -1,
):
    gc.collect()
    torch.cuda.empty_cache()

    # Check if checkpoint paths are empty or do not exist
    if image_checkpoint_path == "":
        raise ValueError("Image checkpoint path not provided")
    if not os.path.exists(image_checkpoint_path):
        raise FileNotFoundError(f"{image_checkpoint_path} does not exist")
    if audio_checkpoint_path == "":
        raise ValueError("Audio checkpoint path not provided")
    if not os.path.exists(audio_checkpoint_path):
        raise FileNotFoundError(f"{audio_checkpoint_path} does not exist")

    # Get source dataset names
    _, img_src_ds_name, _ = image_checkpoint_path.split("/")
    _, audio_src_ds_name, _ = audio_checkpoint_path.split("/")

    # Experiment info
    print("SMITH-WATERMAN LATE FUSION TEST EXPERIMENT")
    print(f"\tImage source dataset: {img_src_ds_name}")
    print(f"\tAudio source dataset: {audio_src_ds_name}")
    print(f"\tTest dataset: {ds_name}")
    print(f"\tKern encoding: {krn_encoding}")
    print(f"\tUse distorted images: {use_distorted_images}")
    print(f"\tImage height: {img_height}")
    print(f"\tImage model checkpoint path: {image_checkpoint_path}")
    print(f"\tAudio model checkpoint path: {audio_checkpoint_path}")

    # Update wandb config
    wandb_logger = WandbLogger(
        project="OMR-A2S-Poly-Multimodal",
        group="SMITH-WATERMAN-LATE-FUSION",
        name=f"{krn_encoding}_ImageTrain-{img_src_ds_name}-AudioTrain-{audio_src_ds_name}_Test-{ds_name}",
        log_model=False,
        entity="grfia",
    )
    wandb_logger.experiment.config.update(
        {
            "img_checkpoint_path": image_checkpoint_path,
            "audio_checkpoint_path": audio_checkpoint_path,
            "match": match,
            "mismatch": mismatch,
            "gap_penalty": gap_penalty,
            "krn_encoding": krn_encoding,
            "use_distorted_images": use_distorted_images,
            "img_height": img_height,
        }
    )

    # Freeze models and put them in eval mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_model = Transformer.load_from_checkpoint(image_checkpoint_path, map_location=device).to(device)
    audio_model = Transformer.load_from_checkpoint(audio_checkpoint_path, map_location=device).to(device)
    image_model.freeze()
    audio_model.freeze()
    image_model.eval()
    audio_model.eval()

    # Get test data loader and i2w dictionary
    datamodule = ARDataModule(
        ds_name=ds_name,
        krn_encoding=krn_encoding,
        input_modality="both",
        use_distorted_images=use_distorted_images,
        img_height=img_height,
    )
    datamodule.setup(stage="test")
    ytest_i2w = datamodule.test_ds.i2w
    test_loader = datamodule.test_dataloader()

    ################################################ FIRST PART: OBTAIN INDIVIDUAL PREDICTIONS
    # Iterate over test set
    Y = []
    IMG_YHAT, IMG_YHAT_PROB = [], []
    AUDIO_YHAT, AUDIO_YHAT_PROB = [], []
    with torch.no_grad():
        for batch in track(test_loader, description="Obtaining individual predictions..."):
            xi, xa, y = batch

            # Get image model prediction
            img_yhat, img_yhat_prob = image_model.get_pred_seq_and_pred_prob_seq(xi.to(image_model.device))
            IMG_YHAT.append(img_yhat)
            IMG_YHAT_PROB.append(img_yhat_prob)

            # Get audio model prediction
            audio_yhat, audio_yhat_prob = audio_model.get_pred_seq_and_pred_prob_seq(xa.to(audio_model.device))
            AUDIO_YHAT.append(audio_yhat)
            AUDIO_YHAT_PROB.append(audio_yhat_prob)

            # Decode ground-truth
            y = [ytest_i2w[i.item()] for i in y[0][1:]]  # Remove SOS_TOKEN
            Y.append(y)

    ################################################ SECOND PART: PERFORM SMITH-WATERMAN FUSION
    # Obtain the callable object of swalign library that contains the align() method that performs the alignment
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    # Gap penalty designates scores for insertion or deletion
    sw = swalign.LocalAlignment(scoring, gap_penalty=gap_penalty)
    # Perform the multimodal combination at prediction level
    YHAT = []
    for r, r_prob, q, q_prob in track(zip(
        IMG_YHAT, IMG_YHAT_PROB, AUDIO_YHAT, AUDIO_YHAT_PROB
    ), description="Performing Smith-Waterman fusion..."):
        # Prepare for swalign computation
        r, q, swa2w = swalign_preprocess(r, q)
        # Smith-Waterman local alignment -> ref, query
        alignment = sw.align(r, q)
        q, m, r = dump(alignment)
        # Fusion policy
        q_prob = preprocess_prob(q, q_prob)
        r_prob = preprocess_prob(r, r_prob)
        alignment = get_alignment(q, m, r, q_prob, r_prob)
        # Undo the swalign preprocess and append to accumulator variable
        YHAT.append(undo_swalign_preprocess(alignment, swa2w))

    ################################################ THIRD PART: COMPUTE METRICS
    print("Computing metrics...")
    metrics = compute_metrics(y_true=Y, y_pred=YHAT)

    # Log metrics to wandb
    wandb_logger.log_metrics(metrics, step=0)
    for k, v in metrics.items():
        print(f"\t{k}: {v}")

    # Print random samples
    index = random.randint(0, len(Y) - 1)
    print(f"Ground truth - {Y[index]}")
    print(f"Prediction - {YHAT[index]}")

    print("Done!")


if __name__ == "__main__":
    fire.Fire(test)
