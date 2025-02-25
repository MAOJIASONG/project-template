# import debugpy; debugpy.connect(('localhost', 9501))
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run the KTO training script with the commands below. In general, the optimal configuration for KTO will be similar to that of DPO.

# Full training:
python examples/scripts/kto.py \
    --model_name_or_path=trl-lib/qwen1.5-1.8b-sft \
    --per_device_train_batch_size 16 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir=kto-aligned-model \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --bf16 \
    --logging_first_step

# QLoRA:
python examples/scripts/kto.py \
    --model_name_or_path=trl-lib/qwen1.5-1.8b-sft \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir=kto-aligned-model-lora \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --use_peft \
    --load_in_4bit \
    --lora_target_modules=all-linear \
    --lora_r=16 \
    --lora_alpha=16
"""

import colorlog
import random
import sys
import copy
from contextlib import nullcontext


import torch
from datasets import concatenate_datasets
from tqdm.auto import tqdm
import datasets
import transformers
from accelerate.state import PartialState
from peft import get_peft_model, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    KTOConfig, 
    KTOTrainer, 
    ModelConfig, 
    get_kbit_device_map, 
    get_peft_config,
    get_quantization_config,
    setup_chat_format
)

from alignment import (
    H4ArgumentParser, 
    DataArguments,
    get_datasets,
    is_adapter_model
)

tqdm.pandas()


logger = colorlog.getLogger(__name__)

if __name__ == "__main__":
    parser = H4ArgumentParser((DataArguments, ModelConfig, KTOConfig))
    data_args, model_args, training_args = parser.parse()
        
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    ###############
    # Setup logging
    ###############
    fmt_string = '%(log_color)s %(asctime)s - %(levelname)s - %(message)s'
    log_colors = {
            'DEBUG': 'white',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'purple'
            }
    log_level = training_args.get_process_log_level()
    colorlog.basicConfig(
        log_colors=log_colors, 
        format=fmt_string, 
        handlers=[colorlog.StreamHandler(sys.stdout)],
        level = log_level
    )
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")


    ################
    # Model & Tokenizer
    ################
    # MODEL
    logger.info("*** Loading pretrained model and tokenizer ***")
    
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    
    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision) is True:
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = get_peft_model(base_model, peft_config)
        model_kwargs = None

    ref_model = copy.deepcopy(model)
    ref_model_kwargs = copy.deepcopy(model_kwargs)

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None
    
    # TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "right"
    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side
    tokenizer.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = training_args.max_length
    # If we are aligning a base model, we use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)
    
    if "llama-3" in model_args.model_name_or_path.lower():
        # For llama3 only
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")
    
    
    ################
    # Dataset
    ################
    logger.info("*** Loading datasets ***") 
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["chosen", "rejected", "prompt"],
    )
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    
    
    #####################
    # Apply chat template
    ##################### 
    def formatting_chat_func(example, is_chosen=True):
        prompt = [{"role":"user", "content": example["prompt"]}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        completion = example["chosen"] if is_chosen else example["rejected"]
        return {"prompt": prompt, "completion": completion, "label": is_chosen}
    
    with PartialState().local_main_process_first():
        raw_chosen_datasets = raw_datasets.map(
            lambda e: formatting_chat_func(e, True),
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names if training_args.remove_unused_columns else None,
            desc="Formatting comparisons with prompt template"
        )
        raw_rejected_datasets = raw_datasets.map(
            lambda e: formatting_chat_func(e, False),
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names if training_args.remove_unused_columns else None,
            desc="Formatting comparisons with prompt template"
        )
    
    # Log a few random samples from the training set:
    if PartialState().is_main_process:
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_chosen_datasets['train'][index]['prompt']}")
            logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_chosen_datasets['train'][index]['completion']}")
            logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_rejected_datasets['train'][index]['completion']}")

    train_dataset = concatenate_datasets([raw_chosen_datasets["train"], raw_rejected_datasets["train"]])
    eval_dataset = concatenate_datasets([raw_chosen_datasets["test"], raw_rejected_datasets["test"]])
    
    
    ################
    # Instantiate KTO trainer
    ################
    if training_args.model_init_kwargs is None:
        training_args.model_init_kwargs = model_kwargs
        training_args.ref_model_init_kwargs = ref_model_kwargs
    else:
        training_args.model_init_kwargs.update(model_kwargs)
        training_args.ref_model_init_kwargs.update(ref_model_kwargs)
    trainer = KTOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    
    ###############
    # Training loop
    ###############
    logger.info("*** Training ***")
    checkpoint = None
    # Check for last checkpoint
    if training_args.resume_from_checkpoint is not None:
        checkpoint = get_last_checkpoint(training_args.output_dir) if isinstance(training_args.resume_from_checkpoint, bool) else training_args.resume_from_checkpoint
        if checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training at {checkpoint=}.")
        else:
            logger.error(f"Failed to load last checkpoint at {checkpoint=}. Start training from scratch")
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("*** Training complete ***")
     
     
    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    
    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
        
    
    ##########
    # Evaluate
    ##########
    torch.cuda.empty_cache()
    if training_args.do_eval:
        logger.info("*** Evaluating ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("*** Evaluating complete ***")
