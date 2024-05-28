______________________________________________________________________

<div align="center">

# Heart Segmentation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>


</div>

## Code Structure

The codebase is structured clearly, with self-explanatory folder names. To assist you further, here are some specific details:

- **Dataset Placement**:
  - Place the dataset (heart) into the following directories:
    - `/data/heart/golds`
    - `/data/heart/images`
  
- **UNet Model**:
  - The UNet model is located at `/src/models/components/unet.py`.
  - The model closely follows the original UNet paper, with an added padding of 1 in the convolutional blocks to handle input and output image sizes.

- **Overall Network Module**:
  - The main network module can be found at `/src/models/heart_module`.
  - This module also includes evaluation metrics, incorporating `torchmetrics` for evaluation:
    - Evaluation is handled as requested, taking the average across the batch dimension. For more details, refer to the [Torchmetrics F1 Score documentation](https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html).

- **Dataset and Dataloader**:
  - Located in `/src/data` and `/src/data/components`.

- **Visualization**:
  - For visualization, please check `visualize_samples.ipynb` located under `/notebooks`.

- **Experiments**:
  - All experiment configurations can be found under `/configs/experiment`.


## Installation

#### Pip

```bash
# clone project or unzip
git clone ...
cd heart_segmentation

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project or unzip
git clone ...
cd heart_segmentation

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=heart_decreased_depth.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
