# Mixtral 8x7B
[Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts/) is a high-quality sparse mixture of experts (MoE) model that matches or beats GPT3.5 on most benchmarks. This repro is a simple and efficient PyTorch native implementation of Mixtral 8x7B.

## Downloading Weights

```bash
export MODEL_REPO=mistralai/Mixtral-8x7B-v0.1
python scripts/download.py --repo_id $MODEL_REPO
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$MODEL_REPO
```

## Generate Text

Model definition in `model.py`, generation code in `generate.py`.

```bash
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"
```

To squeeze out a little bit more performance, you can also compile the prefill with `--compile_prefill`. This will increase compilation times though.

## Quantization
### Int8 Weight-Only Quantization
To generate this version of the model
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int8.pth
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int8
```
To run with int8, just pass the int8 checkpoint to generate.py.
```bash
python generate.py --compile --compile_prefill --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth
```

## Tensor Parallelism
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 generate.py --compile --compile_prefill --checkpoint_path checkpoints/$MODEL_REPO/model.pth
```
