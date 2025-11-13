#!/usr/bin/env bash

# run model ops in fp32, fp16 and bf16
printf "########## Running model ops with fp32 type ###########\n"
python benchmark_xpu.py --run-model-ops --dtype fp32 |& tee model_ops_fp32_xpu.log
printf "\n########## Running model ops with fp16 type ###########\n"
python benchmark_xpu.py --run-model-ops --dtype fp16 |& tee model_ops_fp16_xpu.log
printf "\n########## Running model ops with bf16 type ###########\n"
python benchmark_xpu.py --run-model-ops --dtype bf16 |& tee model_ops_bf16_xpu.log

# run predefined ops with generic tensor size of 64-1024-1024
printf "\n########## Running pre-defined ops with fp32 type ###########\n"
python benchmark_xpu.py --run-predefined --dtype fp32 |& tee predefined_ops_fp32_xpu.log
printf "\n########## Running pre-defined ops with fp16 type ###########\n"
python benchmark_xpu.py --run-predefined --dtype fp16 |& tee predefined_ops_fp16_xpu.log
printf "\n########## Running pre-defined ops with bf16 type ###########\n"
python benchmark_xpu.py --run-predefined --dtype bf16 |& tee predefined_ops_bf16_xpu.log

printf "Done\n"

