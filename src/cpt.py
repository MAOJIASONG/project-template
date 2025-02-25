# import debugpy; debugpy.connect(('localhost', 9501))
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""

import colorlog
import random
import sys

import datasets
import torch
import transformers

from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers.trainer_utils import get_last_checkpoint, EvalPrediction
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map
)

from alignment import (
    H4ArgumentParser, 
    DataArguments,
    get_datasets
)

tqdm.pandas()


logger = colorlog.getLogger(__name__)

if __name__ == "__main__":
    parser = H4ArgumentParser((DataArguments, ModelConfig, SFTConfig))
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
    # TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=True, trust_remote_code=model_args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side
    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = training_args.max_seq_length
    assert tokenizer.chat_template is not None, "Needs chat template!"


    ################
    # Dataset
    ################
    logger.info("*** Loading datasets ***")
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label", "labels"],
    )
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    #####################
    # Apply chat template
    #####################    
    def formatting_chat_func(examples):
        output_texts = tokenizer.apply_chat_template(examples["messages"], tokenize=False, add_generation_prompt=False)
        return output_texts
    

    #########################
    # Define evaluate metric
    #########################
    def compute_metrics(eval_pred: EvalPrediction):
        shift_logits = eval_pred.predictions[..., :-1, :]
        shift_labels = eval_pred.label_ids[..., 1:]
        batch_size, seq_length, vocab_size = shift_logits.shape
        # Flatten the tokens
        perplexity = torch.exp(torch.nn.functional.cross_entropy(torch.tensor(shift_logits).view(batch_size * seq_length, vocab_size), torch.tensor(shift_labels).view(batch_size * seq_length)))

        return {"perplexity": perplexity.item()}
    
    class PrintMetricsCallback(TrainerCallback):
        def on_evaluate(self, args:TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            # 仅在主进程打印
            if state.is_world_process_zero:
                logs = state.log_history[-1]
                if "eval_perplexity" in logs:
                    logger.critical(f"Perplexity: {logs['eval_perplexity']}")


    ################
    # Initialize the Trainer
    ################
    if training_args.model_init_kwargs is None:
        training_args.model_init_kwargs = model_kwargs
    else:
        training_args.model_init_kwargs.update(model_kwargs)
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
        formatting_func=formatting_chat_func,
        compute_metrics=compute_metrics,
        callbacks=[PrintMetricsCallback]
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
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

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