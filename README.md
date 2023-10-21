# Multimodal audio and image to score transcription

## Data downloading and preparation

- Download the GRANDSTAFF dataset from here: [https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz](https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz)
- Put the downloaded files in the data folder (data/grandstaff/...)
- Download a sound font in .sf2 format and update the path in the [data/krn2audio.py](data/krn2audio.py) script
- Run the [prepare_dataset.sh](prepare_dataset.sh) script to create the wav files and the partitions.

## Setup

Install the required libraries:

```bash
pip install -r requirements
```

Alternatively, you can use the included Dockerfile:
```bash
docker build --tag omr_a2s_multimodal_transformer:latest .
```
