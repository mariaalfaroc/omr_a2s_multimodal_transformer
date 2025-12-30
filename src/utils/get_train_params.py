import os
from contextlib import redirect_stdout
from functools import wraps

from src.transformer.model import MultimodalTransformer, Transformer
from src.data.ar_dataset import ARDataModule


def silence_prints(enabled: bool = True):
    """Decorator to optionally silence all prints inside a function."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                # Normal behavior
                return func(*args, **kwargs)

            with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def count_trainable_parameters(model) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@silence_prints(enabled=True)
def build_datamodule(input_modality: str) -> ARDataModule:
    """Instantiate and set up the ARDataModule."""
    datamodule = ARDataModule(
        ds_name="grandstaff",
        krn_encoding="kern",
        input_modality=input_modality,
        use_distorted_images=True,
        img_height=None,
        batch_size=1,
    )
    datamodule.setup(stage="fit")
    return datamodule


@silence_prints(enabled=True)
def build_model(input_modality: str, mixer_type: str | None, datamodule: ARDataModule):
    """Build the appropriate model (unimodal or multimodal)."""
    w2i, i2w = datamodule.get_w2i_and_i2w()
    max_seq_len = datamodule.get_max_seq_len()

    if input_modality == "both":
        (max_h_img, max_w_img), (max_h_audio, max_w_audio) = datamodule.get_max_input_size()
        return MultimodalTransformer(
            max_img_height=max_h_img,
            max_img_width=max_w_img,
            max_audio_height=max_h_audio,
            max_audio_width=max_w_audio,
            max_seq_len=max_seq_len,
            w2i=w2i,
            i2w=i2w,
            mixer_type=mixer_type,
            attn_window=100,
            teacher_forcing_prob=0.2,
            teacher_forcing_modality_prob=0.2,
        )

    max_h, max_w = datamodule.get_max_input_size()
    return Transformer(
        max_input_height=max_h,
        max_input_width=max_w,
        max_seq_len=max_seq_len,
        w2i=w2i,
        i2w=i2w,
        attn_window=100,
        teacher_forcing_prob=0.2,
    )


def main() -> None:
    input_modalities = ["audio", "image", "both"]
    mixer_types = ["concat", "attn_img", "attn_audio", "attn_both"]

    for input_modality in input_modalities:
        for mixer_type in mixer_types if input_modality == "both" else [None]:
            print("=" * 100)
            print(f"INPUT MODALITY: {input_modality}")
            if input_modality == "both":
                print(f"MIXER TYPE: {mixer_type}")

            datamodule = build_datamodule(input_modality)
            model = build_model(input_modality, mixer_type, datamodule)

            num_params = count_trainable_parameters(model)
            print(f"NUMBER OF TRAINABLE PARAMS: {num_params}")


if __name__ == "__main__":
    main()
