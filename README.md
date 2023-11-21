# Installation

## Dependencies

Set up python with pyenv

```
sudo apt install zlib1g zlib1g-dev libssl-dev libbz2-dev libsqlite3-dev
sudo apt install liblzma-dev libncurses5-dev libffi-dev libreadline-dev
pyenv install 3.12.0
pyenv shell 3.12.0
```

Install dependencies

```
pip install equinox
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install transformers safetensors sentencepiece jmp
```

## Download models

Install Git LFS if not installed:

```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install git-lfs
```

Download LLaMA 2 7B Chat:

```
git config --global credential.helper store
huggingface-cli login
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf models/Llama-2-7b-hf
```

## Install pprof

Required for memory usage checking.

Install Go if not installed:

```
wget https://go.dev/dl/go1.21.4.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.21.4.linux-amd64.tar.gz
```

Install pprof:

```
go install github.com/google/pprof@latest
echo '\n\nexport PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
```