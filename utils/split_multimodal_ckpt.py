import os
from copy import deepcopy

import fire
import torch


def split_both_ckpt_in_two(ckpt_path: str, device_name: str = "cpu"):
    def modify_model_checkpoint_callback(ckpt: dict, modality: str = "image_distorted"):
        ckpt_key = [k for k in ckpt["callbacks"].keys() if "ModelCheckpoint" in k][0]
        org_path = ckpt["callbacks"][ckpt_key]["best_model_path"]
        new_path = (
            os.path.splitext(org_path)[0]
            + f"_only_{modality}"
            + os.path.splitext(org_path)[1]
        )
        ckpt["callbacks"][ckpt_key]["best_model_path"] = new_path
        ckpt["callbacks"][ckpt_key]["kth_best_model_path"] = new_path
        ckpt["callbacks"][ckpt_key]["best_k_models"] = {
            new_path: ckpt["callbacks"][ckpt_key]["best_model_score"]
        }
        return ckpt

    def modify_hyper_parameters(ckpt: dict, modality: str = "image"):
        del ckpt["hyper_parameters"]["mixer_type"]
        del ckpt["hyper_parameters"]["teacher_forcing_modality_prob"]
        if modality == "image":
            del ckpt["hyper_parameters"]["max_audio_height"]
            del ckpt["hyper_parameters"]["max_audio_width"]

            ckpt["hyper_parameters"]["max_input_height"] = ckpt["hyper_parameters"][
                "max_img_height"
            ]
            ckpt["hyper_parameters"]["max_input_width"] = ckpt["hyper_parameters"][
                "max_img_width"
            ]

            del ckpt["hyper_parameters"]["max_img_height"]
            del ckpt["hyper_parameters"]["max_img_width"]
        elif modality == "audio":
            del ckpt["hyper_parameters"]["max_img_height"]
            del ckpt["hyper_parameters"]["max_img_width"]

            ckpt["hyper_parameters"]["max_input_height"] = ckpt["hyper_parameters"][
                "max_audio_height"
            ]
            ckpt["hyper_parameters"]["max_input_width"] = ckpt["hyper_parameters"][
                "max_audio_width"
            ]

            del ckpt["hyper_parameters"]["max_audio_height"]
            del ckpt["hyper_parameters"]["max_audio_width"]
        else:
            raise ValueError(f"Unknown modality: {modality}")
        return ckpt

    def modify_state_dict(ckpt: dict, modality: str = "image"):
        def remove_weights(ckpt: dict, modality: str = "image"):
            keys_to_remove = [
                k
                for k in ckpt["state_dict"].keys()
                if k.startswith(f"{modality}_encoder")
                or k.startswith(f"{modality}_pos_2d")
                or k.startswith("cross_attn")
            ]
            for key in keys_to_remove:
                del ckpt["state_dict"][key]
            return ckpt

        def remove_modality_prefix(ckpt: dict, modality: str = "image"):
            org_keys = list(ckpt["state_dict"].keys())
            for key in org_keys:
                if key.startswith(f"{modality}_"):
                    new_key = key.replace(f"{modality}_", "", 1)
                    ckpt["state_dict"][new_key] = ckpt["state_dict"].pop(key)
            return ckpt

        if modality == "image":
            ckpt = remove_weights(ckpt, modality="audio")
            ckpt = remove_modality_prefix(ckpt, modality="image")
        elif modality == "audio":
            ckpt = remove_weights(ckpt, modality="image")
            ckpt = remove_modality_prefix(ckpt, modality="audio")
        else:
            raise ValueError(f"Unknown modality: {modality}")
        return ckpt

    def save_model(ckpt: dict, path: str):
        torch.save(ckpt, path)

    # Multimodal model
    if "cuda" in device_name:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print("CUDA is not available. Using CPU.")
            device = torch.device("cpu")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device: {device_name}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # NOTE:
    # MixedPrecision 'scale' key is different for audio and image
    # Audio == 8192.0; Image == 4096.0
    # For now, do not modify this key

    # Image model
    img_model = deepcopy(ckpt)
    img_model = modify_model_checkpoint_callback(img_model, modality="image_distorted")
    img_model = modify_hyper_parameters(img_model, modality="image")
    img_model = modify_state_dict(img_model, modality="image")
    img_ckp_path = (
        os.path.splitext(ckpt_path)[0]
        + "_only_image_distorted"
        + os.path.splitext(ckpt_path)[1]
    )
    save_model(img_model, img_ckp_path)

    # Audio model
    audio_model = deepcopy(ckpt)
    audio_model = modify_model_checkpoint_callback(audio_model, modality="audio")
    audio_model = modify_hyper_parameters(audio_model, modality="audio")
    audio_model = modify_state_dict(audio_model, modality="audio")
    audio_ckp_path = (
        os.path.splitext(ckpt_path)[0] + "_only_audio" + os.path.splitext(ckpt_path)[1]
    )
    save_model(audio_model, audio_ckp_path)

    print(f"Image model saved at: {img_ckp_path}")
    print(f"Audio model saved at: {audio_ckp_path}")


if __name__ == "__main__":
    fire.Fire(split_both_ckpt_in_two)
