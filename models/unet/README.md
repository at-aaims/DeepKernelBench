## Quick start

The PyTorch training code is essentially taken from the [guided-diffusion](https://github.com/openai/guided-diffusion) repo. To run PyTorch training, do:

### On NVIDIA/AMD GPUs
```bash
python train.py --data_dir data/birds --iteration 10 --log_interval 10 # use --compile 0 if you don't want to call torch.compile() on the model
```

### On Intel GPUs
```bash
python train_xpu.py --data_dir data/birds --iteration 10 --log_interval 10 # use --compile 0 if you don't want to call torch.compile() on the model
```

