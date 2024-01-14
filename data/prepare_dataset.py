import os
import requests
import tarfile
import shutil

from music21 import converter
from midi2audio import FluidSynth


GRANDSTAFF_PATH = "./grandstaff"
SOUND_FONT = "./data/SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2"


def download_and_extract_grandstaff_dataset():
    file_path = "grandstaff.tgz"
    extract_path = GRANDSTAFF_PATH
    os.makedirs(extract_path, exist_ok=True)

    # Download dataset
    response = requests.get(url="https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz")
    with open(file_path, "wb") as file:
        file.write(response.content)
    # Extract dataset
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(extract_path)
    # Remove tar file
    # os.remove(file_path)


def parse_grandstaff_dataset():
    for composer in os.listdir(GRANDSTAFF_PATH):
        composer_path = os.path.join(GRANDSTAFF_PATH, composer)

        # Create a new folder structure for the composer
        os.makedirs(os.path.join(composer_path, "wav"), exist_ok=True)
        os.makedirs(os.path.join(composer_path, "krn"), exist_ok=True)
        os.makedirs(os.path.join(composer_path, "bekrn"), exist_ok=True)
        os.makedirs(os.path.join(composer_path, "img"), exist_ok=True)
        os.makedirs(os.path.join(composer_path, "img_distorted"), exist_ok=True)

        # Move files to the new folder structure
        for foldername, subfolders, filenames in os.walk(composer_path):
            for filename in filenames:
                if filename.startswith("."):
                    continue
                if filename.endswith(".bekrn"):
                    shutil.move(
                        os.path.join(foldername, filename),
                        os.path.join(composer_path, "bekrn", filename),
                    )
                elif filename.endswith(".krn"):
                    shutil.move(
                        os.path.join(foldername, filename),
                        os.path.join(composer_path, "krn", filename),
                    )
                elif filename.endswith(".jpg"):
                    shutil.move(
                        os.path.join(foldername, filename),
                        os.path.join(composer_path, "img", filename),
                    )
                elif filename.endswith("_distorted.jpg"):
                    shutil.move(
                        os.path.join(foldername, filename),
                        os.path.join(composer_path, "img_distorted", filename),
                    )
                else:
                    continue

        # Remove the remaining folders and files
        for f in os.listdir(composer_path):
            if f not in ["wav", "krn", "bekrn", "img", "img_distorted"]:
                shutil.rmtree(os.path.join(composer_path, f))


def krn2wav():
    os.makedirs("./grandstaff/errors", exist_ok=True)

    fs = FluidSynth(sample_rate=22050, sound_font=SOUND_FONT)
    for composer in os.listdir(GRANDSTAFF_PATH):
        print(f"Generating wav files for {composer}")
        composer_path = os.path.join(GRANDSTAFF_PATH, composer)

        errors = []
        for id, krn_file in enumerate(os.listdir(os.path.join(composer_path, "krn"))):
            # krn to midi
            try:
                krn_stream = converter.parse(
                    os.path.join(composer_path, "krn", krn_file)
                )
            except Exception as err:
                errors.append(krn_file + "\t" + type(err) + "\t" + str(err))
                # Remove the file and all its corresponding files
                os.remove(os.path.join(composer_path, "krn", krn_file))
                os.remove(
                    os.path.join(
                        composer_path,
                        "img",
                        krn_file.replace(".krn", ".jpg"),
                    )
                )
                os.remove(
                    os.path.join(
                        composer_path,
                        "img_distorted",
                        krn_file.replace(".krn", "_distorted.jpg"),
                    )
                )
                os.remove(
                    os.path.join(
                        composer_path,
                        "bekrn",
                        krn_file.replace(".krn", ".bekrn"),
                    )
                )
                continue
            midi_file = os.path.join(composer_path, "krn", krn_file + ".mid")
            _ = krn_stream.write("midi", fp=midi_file)

            # midi to wav
            wav_file = os.path.join(
                composer_path, "wav", krn_file.replace(".krn", ".wav")
            )
            fs.midi_to_audio(midi_file, wav_file)
            os.remove(midi_file)

        # Save errors
        print(f"{len(errors)} out of {id+1} files could not be converted to wav.")
        errors = "\n".join(errors)
        with open(os.path.join("./grandstaff/errors", "composer.txt"), "w") as f:
            f.write(errors)
        print(f"Errors saved to ./grandstaff/errors/{composer}.txt")


if __name__ == "__main__":
    download_and_extract_grandstaff_dataset()
    parse_grandstaff_dataset()
    krn2wav()
