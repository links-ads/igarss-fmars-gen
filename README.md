# FMARS: Annotating Remote Sensing Images for Disaster Management using Foundation Models

Dataset and code for generating the dataset described in the paper *FMARS: Annotating Remote Sensing Images for Disaster Management using Foundation Models*.

![FMARS workflow](annotations-flow.png)

[![arXiv](https://img.shields.io/badge/arXiv-2405.20109-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2405.20109)

> **Note:** The dataset is available at [Hugging Face Datasets](https://huggingface.co/datasets/links-ads/fmars-dataset).

## Environment Setup

### Python Virtual Environment

We recommend using Python version 3.10.12 for this project. Set up a new virtual environment using the following commands:

```shell
pyenv install 3.10.12
pyenv global 3.10.12
python -m venv .venv
source .venv/bin/activate
```

Install the required Python packages with:
```shell
pip install -r requirements.txt
```
### Resources
Download necessary metadata using:
```shell
wget -q https://github.com/links-ads/igarss-fmars-gen/releases/download/v0.1.0/metadata.zip
unzip metadata.zip
rm metadata.zip
```
### Model Weights

Download the model weights for **Grounding DINO** and **EfficientSAM**:

- **Grounding DINO**:
```shell
cd models/GDINO/weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../../..
```

- **EfficientSAM**:
  Download the checkpoint from [this link](https://github.com/yformer/EfficientSAM/blob/main/weights/efficient_sam_vitt.pt) and place it in `models/efficientSAM/weights/`.

### Data Acquisition

- **Maxar Images**:

**⚠️ WARNING:** Downloading all necessary Maxar images requires more than 900 GB of free space and several hours.
```shell
python src/maxarseg/scripts/downloadMaxar.py
```

- **Microsoft Road Detections**:
```shell
python src/maxarseg/scripts/downloadRoads.py
```
## Usage

To run the automatic labelling process on a whole event:
```shell
python -B -O -m maxarseg.main_seg_event_w_config --out_dir_root <path/to/output/directory> \
        --config "./configs/trees_cfg.yaml" \
        --event_ix 0
```
To label one-tenth of the event (partitioned processing):
```shell
python -B -O -m maxarseg.main_seg_event_w_config_partitioned --out_dir_root <path/to/output/directory> \
        --config "./configs/trees_cfg.yaml" \
        --event_ix 0.1
```
Replace 0.1 with values from x.0 to x.9 to process different partitions.
## Acknowledgements

FMARS leverages several open-source projects. We extend our gratitude to the authors of the following tools for making their software publicly available:

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [EfficientSAM](https://github.com/yformer/EfficientSAM)