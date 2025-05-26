<p align='center'>
  <a href=''><img src='https://i.imgur.com/Iu7CvC1.png' alt='PRAIG-logo' width='100'></a>
</p>

<h1 align='center'>Multimodal audio and image to score transcription</h1>

<h4 align='center'>Full text coming soon<a href='' target='_blank'></a>.</h4>

<p align='center'>
  <img src='https://img.shields.io/badge/python-3.11.0-orange' alt='Python'>
  <img src='https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white' alt='PyTorch'>
  <img src='https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white' alt='Lightning'>
  <img src='https://img.shields.io/static/v1?label=License&message=MIT&color=blue' alt='License'>
</p>

<p align='center'>
  <a href='#about'>About</a> •
  <a href='#how-to-use'>How To Use</a> •
  <a href='#citations'>Citations</a> •
  <a href='#acknowledgments'>Acknowledgments</a> •
  <a href='#license'>License</a>
</p>


## About

*Coming soon.*


## How To Use

### Set Up with Conda

Follow these steps to set up the environment using Conda:

```bash
# Clone the repository
git clone https://github.com/mariaalfaroc/omr_a2s_multimodal_transformer

# Navigate to the project directory
cd omr_a2s_multimodal_transformer

# Create and activate a new Conda environment
conda create -n omr_a2s_transformer python=3.11
conda activate omr_a2s_transformer

# Install the required packages
pip install -r requirements.txt
```

#### Set Up with Docker

Follow these steps to set up the project using Docker:

```bash
# Clone the repository
git clone https://github.com/mariaalfaroc/omr_a2s_multimodal_transformer

# Navigate to the project directory
cd omr_a2s_multimodal_transformer

# Build the Docker image
docker build -t omr_a2s_transformer_image .

# Run (launch) the Docker container
docker run --name omr_a2s_transformer --gpus "device=0" -itd --rm --shm-size=40g -v $(pwd):/app omr_a2s_transformer_image

# Enter the running Docker container’s shell
docker exec -it omr_a2s_transformer /bin/bash

# Install the required packages inside the container
pip install -r requirements.txt
```

### Dataset

We use the [**GRANDSTAFF**](https://sites.google.com/view/multiscore-project/datasets#h.n7ug4ausi7j) dataset.

The GRANDSTAFF dataset contains 53&nbsp;882 single-system piano scores in common western modern notation, each represented by four files: (i) an image with the rendered score, (i) a distorted image, (iii) the musical symbolic representation of the incipit both in Humdrum **kern format and (iv) in an on-purpose simplified extension of this format called **bekern (*basic extended kern*).

To obtain the corresponding audio files, we must convert the provided Humdrum **kern representations to MIDI and then synthesize the MIDI data using FluidSynth.

The specific steps to follow are:
1) Download a [General MIDI SounFont (sf2)](https://sites.google.com/site/soundfonts4u/#h.p_biJ8J359lC5W). We recommend downloading the [SGM-v2.01 soundfont](https://drive.google.com/file/d/12zSPpFucZXFg-svKeu6dm7-Fe5m20xgJ/view) as this code has been tested using this soundfont. **Place the sf2 file in the [`data`](data) folder.**
2) Run the following script. 
```bash 
$ python -u data/prepare_dataset.py
```

### Experiments

*Coming soon.*


## Citations

```bibtex
@article{luna2025omra2stransformer,
  title     = {{Multimodal Transcription Transformer for Polyphonic Music Transcription}},
  author    = {Luna-Barahona, Noelia and Alfaro-Contreras, Mar{\'\i}a and P{\'\e}rez-Sancho, Carlos and Valero-Mas, Jose J and Calvo-Zaragoza, Jorge},
  journal   = {{}},
  volume    = {},
  pages     = {},
  year      = {2025},
  publisher = {},
  doi       = {},
}
```


## Acknowledgments

This work is part of the I+D+i PID2020-118447RA-I00 ([MultiScore](https://sites.google.com/view/multiscore-project)) project, funded by MCIN/AEI/10.13039/501100011033. Computational resources were provided by the Valencian Government and FEDER funding through IDIFEDER/2020/003.

## License

This work is under a [MIT](LICENSE) license.