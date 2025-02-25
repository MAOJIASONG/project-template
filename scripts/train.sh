#!/bin/bash
export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/accelerate_configs/zero3.yaml \
src/my_app.py \
+exps=qwen2.5-1.5b-sft