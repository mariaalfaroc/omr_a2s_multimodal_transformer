<p align='center'>
  <a href=''><img src='https://i.imgur.com/Iu7CvC1.png' alt='PRAIG-logo' width='100'></a>
</p>

<h1 align='center'>Multimodal audio and image to score transcription</h1>

<!---
<h4 align='center'>Full text coming soon<a href='' target='_blank'></a>.</h4>
--->

<p align='center'>
  <img src='https://img.shields.io/badge/python-3.11.0-orange' alt='Python'>
  <img src='https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white' alt='PyTorch'>
  <img src='https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white' alt='Lightning'>
  <img src='https://img.shields.io/static/v1?label=License&message=MIT&color=blue' alt='License'>
</p>

<p align='center'>
  <!---<a href='#about'>About</a> •--->
  <a href='#how-to-use'>How To Use</a> •
  <!---<a href='#citations'>Citations</a> •--->
  <a href='#acknowledgments'>Acknowledgments</a> •
  <a href='#license'>License</a>
</p>

<!---
## About
--->


## How To Use

### Set up

Install the required [`libraries`](requirements.txt):
```bash
pip install -r requirements
```

Alternatively, you can use the included [`Dockerfile`](Dockerfile):
```bash
docker build --tag omr_a2s_multimodal_transformer:latest .
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

<!---
### Experiments



## Citations

```bibtex
@inproceedings{,
  title     = {{}},
  author    = {},
  booktitle = {{}},
  year      = {},
  publisher = {},
  address   = {},
  month     = {},
}
@article{,
  title     = {{}},
  author    = {},
  journal   = {{}},
  volume    = {},
  pages     = {},
  year      = {},
  publisher = {},
  doi       = {},
}
```
--->

## Acknowledgments

This work is part of the I+D+i PID2020-118447RA-I00 ([MultiScore](https://sites.google.com/view/multiscore-project)) project, funded by MCIN/AEI/10.13039/501100011033. Computational resources were provided by the Valencian Government and FEDER funding through IDIFEDER/2020/003.

## License

This work is under a [MIT](LICENSE) license.