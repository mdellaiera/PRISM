# PRISM
Platform for Research in Imaging and Signal Methodology

```bash
$ conda create -n prism python=3.12
$ conda activate prism
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$ pip install lightning
$ mamba env update -f environment.yml
$ pip install -e .
```