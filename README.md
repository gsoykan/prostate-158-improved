______________________________________________________________________

<div align="center">

# Prostate Anatomy and Tumor Segmentation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>


</div>

## Code Structure

The codebase is structured clearly, with self-explanatory folder names. To assist you further, here are some specific
details:

- **Dataset Placement**:
    - Place the dataset ([Prostate-158](https://github.com/kbressem/prostate158)) into the following directories:
        - `/prostate158/test/**`
        - `/prostate158/train/**`
        - `/prostate158/train.csv`
        - `/prostate158/valid.csv`
        - `/prostate158/test.csv`

- **Overall Network Module**:
    - The main network module can be found at `/src/models/prostate158_module`.
    - This module also includes evaluation metrics, incorporating `monai` for evaluation:
        - Evaluation is handled by taking the average across the batch dimension. For more details, refer to
          the [Monai Metric documentation](https://docs.monai.io/en/stable/metrics.html).

- **Dataset and Dataloader**:
    - Located in `/src/data` and `/src/data/components`.

- **Experiments**:
    - All experiment configurations can be found under `/configs/experiment`.

## Installation

#### Pip

```bash
# clone project or unzip
git clone https://github.com/gsoykan/prostate-158-improved.git
cd prostate-158-improved

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
git clone https://github.com/gsoykan/prostate-158-improved.git
cd prostate-158-improved

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
python src/train.py experiment=tumor.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
