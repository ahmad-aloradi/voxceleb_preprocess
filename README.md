# VoxCeleb Preparation

This repository contains scripts for preparing metadata for the VoxCeleb dataset. The script processes metadata from both VoxCeleb1 and VoxCeleb2, generates training IDs and create utterances metadata.

## Features

- Generate training IDs from combined VoxCeleb1 and VoxCeleb2 metadata.
- Update metadata CSV files with training IDs.
- Load and merge speaker metadata from both VoxCeleb1 and VoxCeleb2.
- Process WAV files and generate utterance metadata.
- Save processed metadata to CSV files.


## Key Files

- `voxceleb_prep.py`: Handles the initial preparation and preprocessing of VoxCeleb data.
- `enrich_veri_test.py`: Enriches the trial file `verit_test.txt`, by including natinalities and genders.


# Requirements
- Python 3.6+
- pandas
- tqdm
- soundfile
- multiprocessing

## Setup

`pip install -r requirements.txt`

## Usage

- For generating the dev file, which contains the training and eval data:
    `python voxceleb_prep.py`

- For generating `veri_test_enriched.csv`, whcih contains the test trials:
    `python enrich_veri_test.py`


## Data

The VoxCeleb dataset files are expected to be located in `data/`. In my case, I have both Voxceleb1 (`dev` & `test`) and Voxceleb2 (`dev`) unpacked in `data/voxceleb`


##  Acknowledgements
This script is based on the VoxCeleb dataset and its metadata. Special thanks to the creators of the VoxCeleb dataset.