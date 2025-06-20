import os
from typing import List, Optional, Tuple

from datasets import DatasetDict, Dataset, Features, Value, Audio, Image
from huggingface_hub import login


def norm_and_case_path(path: str) -> str:
    path = path.replace("\\", os.sep).replace("/", os.sep)
    return os.path.normcase(os.path.normpath(path))


def check_if_file_exists(file_path: str) -> bool:
    file_path_norm = norm_and_case_path(file_path)
    return os.path.exists(file_path_norm) and os.path.isfile(file_path_norm)


def check_if_dir_exists(dir_path: str) -> bool:
    dir_path_norm = norm_and_case_path(dir_path)
    return os.path.exists(dir_path_norm) and os.path.isdir(dir_path_norm)


def remove_duplicate_spaces(text: str) -> str:
    text = " ".join(filter(None, text.split()))
    text = text.strip()
    return text


def get_inputs_and_transcripts_files(
    base_dir: str, ds_name: str, partition_type: str
) -> Tuple[Tuple[Optional[List[str]], Optional[List[str]]], List[str]]:
    partition_file = os.path.join(base_dir, "partitions", ds_name, partition_type + ".txt")

    data = []
    with open(partition_file, "r") as file:
        for s in file.read().splitlines():
            s = s.strip()
            if not s:
                continue

            if ds_name == "grandstaff":
                # For the grandstaff dataset, we expect the format to be "composer\tpiece"
                composer, s = s.split("\t")
                composer = composer.strip()
                s = s.strip()
            else:
                # For other datasets, we expect the format to be "piece"
                composer = ds_name

            composer_folder_path = os.path.join(base_dir, composer)

            image = os.path.join(composer_folder_path, "img", s + ".jpg")
            image_distorted = os.path.join(composer_folder_path, "img_distorted", s + "_distorted.jpg")
            audio = os.path.join(composer_folder_path, "wav", s + ".wav")
            kern = os.path.join(composer_folder_path, "krn", s + ".krn")
            bekern = os.path.join(composer_folder_path, "bekrn", s + ".bekrn")

            if (
                not check_if_file_exists(image)
                or not check_if_file_exists(image_distorted)
                or not check_if_file_exists(audio)
                or not check_if_file_exists(kern)
                or not check_if_file_exists(bekern)
            ):
                print(f"Missing files for {s}: {image}, {image_distorted}, {audio}, {kern}, {bekern}")
                continue

            transcripts = {"kern": kern, "bekern": bekern}
            for transcript_key, transcript_path in transcripts.items():
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                    if not transcript:
                        print(f"Empty transcript for {s}: {transcript_path}")
                        continue
                    transcripts[transcript_key] = transcript

                # Just for visual debugging purposes:
                # if transcript != remove_duplicate_spaces(transcript):
                #     print(
                #         f"Removing duplicate spaces in transcript for {s}: {transcript_path}"
                #     )
                #     # Print before and after for debugging
                #     print(f"Before: '{transcript}'")
                #     print(f"After: '{remove_duplicate_spaces(transcript)}'")

            data.append(
                {
                    "audio": audio,
                    "image": image,
                    "image_distorted": image_distorted,
                    "kern": transcripts["kern"],
                    "bekern": transcripts["bekern"],
                }
            )

    return data


def upload_grandstaff_to_hf(base_dir: str):
    # Ensure the HF_TOKEN environment variable is set
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set! Please, set it to your Hugging Face token")
    login(token=hf_token)

    # Define dataset splits and names
    splits = ["train", "val", "test"]
    datasets = [
        "grandstaff",
        "beethoven",
        "chopin",
        "hummel",
        "joplin",
        "mozart",
        "scarlatti-d",
    ]

    # Define the base directory for the dataset
    base_dir = norm_and_case_path(base_dir)
    if not check_if_dir_exists(base_dir):
        raise FileNotFoundError(f"Base directory {base_dir} does not exist")

    # Iterate over each dataset
    for ds_name in datasets:
        # Create a DatasetDict for each dataset
        dataset_dict = DatasetDict(
            {
                split: Dataset.from_list(
                    get_inputs_and_transcripts_files(base_dir, ds_name, split),
                    features=Features(
                        {
                            "audio": Audio(),
                            "image": Image(),
                            "image_distorted": Image(),
                            "kern": Value("string"),
                            "bekern": Value("string"),
                        }
                    ),
                )
                for split in splits
            }
        )

        # Push to Hugging Face Hub
        dataset_dict.push_to_hub(f"PRAIG/{ds_name}-grandstaff-multimodal")
        print(f"Upload complete for {ds_name}!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload Grandstaff collection to Hugging Face Hub")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing the Grandstaff collection files",
    )
    args, _ = parser.parse_known_args()

    upload_grandstaff_to_hf(args.base_dir)
