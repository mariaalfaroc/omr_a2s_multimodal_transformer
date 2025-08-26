import sys

sys.path.append("./")

import gc
import os
import random
from typing import Optional

import fire
import torch
from lightning.pytorch.loggers.wandb import WandbLogger
from rich.progress import track

from data.ar_dataset import EOS_TOKEN, SOS_TOKEN, ARDataModule
from transformer.model import Transformer
from utils.metrics import compute_metrics
from utils.seed import seed_everything

seed_everything(42, benchmark=False)

# Set WANDB_API_KEY
with open("wandb_api_key.txt", "r") as f:
    os.environ["WANDB_API_KEY"] = f.read().strip()


def weighted_prediction(
    xi: torch.Tensor,
    xa: torch.Tensor,
    img_model: torch.nn.Module,
    audio_model: torch.nn.Module,
    alpha: float = 0.5,
):
    def get_model_embedding(x: torch.Tensor, model: torch.nn.Module):
        x = x.to(model.device)
        # Encoder
        x = model.encoder(x=x)
        # Prepare for decoder
        # 2D PE + flatten + permute
        x = model.pos_2d(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        return x

    # Inference only supports batch_size = 1
    assert xi.size(0) == 1, "Inference only supports batch_size = 1"

    # Get image and audio embeddings
    xi = get_model_embedding(xi, img_model)
    xa = get_model_embedding(xa, audio_model)

    # Autoregressive decoding with weighted prediction
    yhat = []

    y_in = torch.tensor([img_model.w2i[SOS_TOKEN]]).unsqueeze(0).long().to(xi.device)
    for _ in range(max(img_model.max_seq_len, audio_model.max_seq_len)):
        # Get image output vector probability
        img_y_out_hat = img_model.decoder(tgt=y_in, memory=xi, memory_len=None)
        img_y_out_hat = img_y_out_hat[0, :, -1]  # Last token
        img_y_out_hat = img_y_out_hat.softmax(dim=-1)

        # Get audio output vector probability
        audio_y_out_hat = audio_model.decoder(tgt=y_in, memory=xa, memory_len=None)
        audio_y_out_hat = audio_y_out_hat[0, :, -1]  # Last token
        audio_y_out_hat = audio_y_out_hat.softmax(dim=-1)

        # Weighted prediction
        y_out_hat = alpha * img_y_out_hat + (1 - alpha) * audio_y_out_hat
        y_out_hat_token = y_out_hat.argmax(dim=-1).item()
        y_out_hat_word = img_model.i2w[y_out_hat_token]  # Both models have the same vocabulary
        yhat.append(y_out_hat_word)
        if y_out_hat_word == EOS_TOKEN:
            break

        y_in = torch.cat([y_in, torch.tensor([[y_out_hat_token]]).long().to(xi.device)], dim=1)

    return yhat


def test(
    ds_name: str,
    image_checkpoint_path: str,
    audio_checkpoint_path: str,
    krn_encoding: str = "bekern",
    use_distorted_images: bool = False,
    img_height: Optional[int] = None,  # If None, the original image height is used
    alpha: float = 0.5,
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
    print("WEIGHTED MULTIMODAL TOKEN LATE FUSION TEST EXPERIMENT")
    print(f"\tImage source dataset: {img_src_ds_name}")
    print(f"\tAudio source dataset: {audio_src_ds_name}")
    print(f"\tTest dataset: {ds_name}")
    print(f"\tKern encoding: {krn_encoding}")
    print(f"\tUse distorted images: {use_distorted_images}")
    print(f"\tImage height: {img_height}")
    print(f"\tImage model checkpoint path: {image_checkpoint_path}")
    print(f"\tAudio model checkpoint path: {audio_checkpoint_path}")
    print(f"\tAlpha: {alpha}")

    # Update wandb config
    wandb_logger = WandbLogger(
        project="OMR-A2S-Poly-Multimodal",
        group="WEIGHTED-MULTIMODAL-TOKEN-LATE-FUSION",
        name=f"{krn_encoding}_ImageTrain-{img_src_ds_name}-AudioTrain-{audio_src_ds_name}_Test-{ds_name}",
        log_model=False,
        entity="grfia",
    )
    wandb_logger.experiment.config.update(
        {
            "img_checkpoint_path": image_checkpoint_path,
            "audio_checkpoint_path": audio_checkpoint_path,
            "alpha": alpha,
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

    # Check vocabularies match
    assert image_model.w2i == audio_model.w2i, "Vocabularies do not match"

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

    ################################################ FIRST PART: OBTAIN WEIGHTED PREDICTIONS
    # Iterate over test set
    Y = []
    YHAT = []
    with torch.no_grad():
        for batch in track(test_loader, description="Obtaining weighted predictions..."):
            xi, xa, y = batch

            # Get weighted prediction
            yhat = weighted_prediction(xi, xa, image_model, audio_model, alpha)
            YHAT.append(yhat)

            # Decode ground-truth
            y = [ytest_i2w[i.item()] for i in y[0][1:]]  # Remove SOS_TOKEN
            Y.append(y)

    ################################################ SECOND PART: COMPUTE METRICS
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
