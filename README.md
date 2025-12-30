<p align='center'>
  <a href=''><img src='https://i.imgur.com/Iu7CvC1.png' alt='PRAIG-logo' width='100'></a>
</p>

<h1 align='center'>Multimodal audio and image to score transcription</h1>

<h4 align='center'>Full text coming soon<a href='' target='_blank'></a>.</h4>

<p align='center'>
  <img src='https://img.shields.io/badge/python-3.12.0-orange' alt='Python'>
  <img src='https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white' alt='PyTorch'>
  <img src='https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white' alt='Lightning'>
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-white' alt='HuggingFace'>
  <img src='https://img.shields.io/static/v1?label=License&message=MIT&color=blue' alt='License'>
</p>

<p align="center">
  <strong>GRANDSTAFF Collection</strong>
  </br>
  <a href="https://huggingface.co/collections/PRAIG/omr-a2s-multimodal-grandstaff-68541370a4a8f1b983badbb3">
    <img align="center" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg">
  </a>
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

### Prerequisites

- **Python 3.12+**: Required for the application (managed automatically by `uv`)
- **Git**: For cloning the repository
- **Access to Required APIs**: [Wandb](https://wandb.ai/site/) (see [`.env.template`](.env.template))

### Local setup

#### 1. Clone the repository

```bash
git clone https://github.com/mariaalfaroc/omr_a2s_multimodal_transformer
cd omr_a2s_multimodal_transformer
```

#### 2. Install `uv` package manager

[`uv`](https://docs.astral.sh/uv/) is used for fast and reliable dependency management.

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:
```bash
uv --version
```

#### 3. Create the virtual environment

```bash
uv venv
```

This creates a `.venv` directory with an isolated Python environment.

#### 4. Activate the virtual environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

#### 5. Install dependencies

```bash
uv sync
```

This installs all dependencies defined in `pyproject.toml` (and `uv.lock` if present).

#### 6. Environment configuration

```bash
cp .env.template .env
```

Edit `.env` and fill in the required variables (e.g., Wandb credentials).

> If environment variables are not correctly set, the application will fail at startup due to validation.

### Docker setup

#### 1. Clone the repository

```bash
git clone https://github.com/mariaalfaroc/omr_a2s_multimodal_transformer
cd omr_a2s_multimodal_transformer
```

#### 2. Prepare environment configuration

On the host (before building):

```bash
cp .env.template .env
# edit .env with your Wandb and other required settings
```

#### 3. Build the Docker image

```bash
docker build -t omr_a2s_transformer_image .
```

#### 4. Run the container

With GPU and project directory mounted:

```bash
docker run \
  --name omr_a2s_transformer \
  --gpus "device=0" \
  -itd --rm \
  --shm-size=40g \
  --env-file .env \
  -v $(pwd):/app \
  omr_a2s_transformer_image
```

#### 5. Enter the container

```bash
docker exec -it omr_a2s_transformer /bin/bash
```

**Inside the container, everything is already set up:**
- ✅ Virtual environment created (`.venv`)
- ✅ Dependencies installed (`uv sync --frozen`)
- ✅ Environment activated automatically (`PATH` updated)

**Ready to run:** Your scripts and commands work immediately.

### Dataset

<h3 align='center'>🔔 The dataset is now available on <a href='https://huggingface.co/collections/PRAIG/omr-a2s-multimodal-grandstaff-68541370a4a8f1b983badbb3' target='_blank'>Hugging Face</a>.</h3>

We use the [**GRANDSTAFF**](https://sites.google.com/view/multiscore-project/datasets#h.n7ug4ausi7j) dataset, which contains **53&nbsp;882 single-system piano scores in common western modern notation**. Each score is provided in four formats:

1. A rendered image of the score
2. A distorted version of the image
3. The symbolic musical representation in Humdrum **\*\*kern** format
4. A simplified extension of this format called **\*\*bekern** (*basic extended kern*)

To obtain the corresponding audio files, follow these steps:

1. **Download a [General MIDI SoundFont (.sf2)](https://sites.google.com/site/soundfonts4u/#h.p_biJ8J359lC5W)**

    We recommend the [SGM-v2.01 SoundFont](https://drive.google.com/file/d/12zSPpFucZXFg-svKeu6dm7-Fe5m20xgJ/view), which is compatible with our code. **Place the `.sf2` file in the [`data`](src/data) folder.**

2. **Run the dataset preparation script:**

    This script will convert the provided Humdrum **\*\*kern** representations to MIDI and then synthesize the MIDI data using FluidSynth.

    > ⚠️ You **do not need to run the script** if you just want to use the final dataset — it is now directly available on Hugging Face.

    ```bash
    python -u src/data/prepare_dataset.py
    ```

### Experiments

*Coming soon.*


## Citations

```bibtex
@article{luna2025omra2stransformer,
  title     = {{Multimodal Transcription Transformer for Polyphonic Music Transcription}},
  author    = {Alfaro-Contreras, Mar{\'\i}a and Luna-Barahona, Noelia and P{\'\e}rez-Sancho, Carlos and Valero-Mas, Jose J and Calvo-Zaragoza, Jorge},
  journal   = {{}},
  volume    = {},
  pages     = {},
  year      = {2026},
  publisher = {},
  doi       = {},
}
```


## Acknowledgments

This work is part of the I+D+i PID2020-118447RA-I00 ([MultiScore](https://sites.google.com/view/multiscore-project)) project, funded by MCIN/AEI/10.13039/501100011033. Computational resources were provided by the Valencian Government and FEDER funding through IDIFEDER/2020/003.

## License

This work is under a [MIT](LICENSE) license.
