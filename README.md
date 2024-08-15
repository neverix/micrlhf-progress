# What is this
This is a library that contains a standard implementation of LLaMA in [Penzai](https://github.com/google-deepmind/penzai). So far, the library can load LLaMA with 8-bit quantization from GGUF files and run them. In the future, I am planning to implement Paged Attention as well as kernels for 4-bit and 6-bit quantized inference.

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
poetry install
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

Download models:

```
mkdir -p models
# wget -c 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf?download=true' -c -O models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
wget -c 'https://huggingface.co/PrunaAI/Meta-Llama-Guard-2-8B-GGUF-smashed/resolve/main/Meta-Llama-Guard-2-8B.Q8_0.gguf?download=true' -O models/llama-guard-q8_0.gguf
# wget -c 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf?download=true' -O models/phi-3-16.gguf
# wget -c 'https://huggingface.co/lmstudio-community/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf?download=true' -O models/phi-3-16.gguf
wget -c 'https://huggingface.co/SanctumAI/Phi-3-mini-4k-instruct-GGUF/resolve/main/phi-3-mini-4k-instruct.fp16.gguf?download=true' -O models/phi-3-16.gguf
wget -c 'https://huggingface.co/failspy/kappa-3-phi-3-4k-instruct-abliterated-GGUF/resolve/main/ggml-model-f16.gguf?download=true' -O models/abl.gguf
wget -c 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true' -O models/tinyllama-1.1b-q8_0.gguf
wget -c 'https://huggingface.co/mlabonne/gemma-2b-GGUF/resolve/main/gemma-2b.Q8_0.gguf?download=true' -O models/gemma-2b.gguf
wget -c 'https://huggingface.co/mlabonne/gemma-2b-it-GGUF/resolve/main/gemma-2b-it.Q8_0.gguf?download=true' -O models/gemma-2b-it.gguf
wget -c 'https://huggingface.co/mlabonne/gemma-2b-it-GGUF/resolve/main/gemma-2b-it.Q6_K.gguf?download=true' -O models/gemma-2b-it-q6_k.gguf
wget -c 'https://huggingface.co/mlabonne/gemma-2b-it-GGUF/resolve/main/gemma-2b-it.Q4_K_S.gguf' -O models/gemma-2b-it-q4_k_s.gguf
wget -c 'https://huggingface.co/mlabonne/gemma-2b-it-GGUF/resolve/main/gemma-2b-it.Q4_K_M.gguf?download=true' -O models/gemma-2b-it-q4_k.gguf
wget -c 'https://huggingface.co/MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct.Q8_0-00001-of-00002.gguf?download=true' -O models/llama-3-70b-1.gguf
wget -c 'https://huggingface.co/MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct.Q8_0-00002-of-00002.gguf?download=true' -O models/llama-3-70b-2.gguf
wget -c 'https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q8_0.gguf?download=true' -O models/gemma-2-9b-it.gguf
wget -c 'https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q8_0.gguf?download=true' -O models/gemma-2-2b-it.gguf
```

For evals:
```
mkdir -p data/eval_source_data
wget -c https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv -O data/adv.csv
wget -c https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/raw/main/data/harmful-behaviors.csv -O data/jail.csv
wget -c https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/raw/main/data/benign-behaviors.csv -O data/benign.csv
git clone https://github.com/anthropics/evals.git data/eval_source_data/anthropic_evals
wget -c 'https://people.eecs.berkeley.edu/~hendrycks/data.tar' -O data/mmlu.tar
mkdir -p data/sycophancy
wget -c 'https://github.com/meg-tong/sycophancy-eval/raw/main/datasets/answer.jsonl' -O data/sycophancy/answer.jsonl
wget -c 'https://github.com/meg-tong/sycophancy-eval/raw/main/datasets/feedback.jsonl' -O data/sycophancy/feedback.jsonl
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
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
```
