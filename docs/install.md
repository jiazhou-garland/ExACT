# Install

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for environment setup:

```
conda create --name ECLIP python==3.8.0
conda activate ECLIP
```

Then install PyTorch which is compatible with your CUDA setting. In our experiments, we use PyTorch 1.7.1 + CUDA 11.0

```
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
```

Install CLIP from OpenAI
```
pip install ftfy regex tqdm  # packages required by CLIP
pip install git+https://github.com/openai/CLIP.git 
```
If you meet error for install CLIP, try to download it from git or offline then install it manually:
```
git clone https://github.com/openai/CLIP.git
cd CLIP
pip install .
```
Install additional packages:
```
conda install IPython matplotlib  # additional package from conda
pip install einops wandb opencv-python spikingjelly  # additional package from pip
```

We use [wandb](https://wandb.ai/) for logging, please run `wandb login` to log in and `wandb offline` to disable it.
