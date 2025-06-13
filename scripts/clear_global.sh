#! /bin/bash

# clear global_step* in target dir
TARGET_DIR="grpo-qwen25-3b"
find $TARGET_DIR -type d -name "global_step*" -exec rm -rf {} +