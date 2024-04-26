# What is this
This is a library that contains a standard implementation of LLaMA in [Penzai](https://github.com/google-deepmind/penzai). So far, the library can load LLaMA with 8-bit quantization from GGUF files and run them without special kernels. In the future, I am planning to implement Flash and Paged attention as well as kernels for quantized inference.

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
poetry instal
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

Download LLaMA 3 8B Instruct (quantized):

```
wget 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf?download=true' -c -O models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
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