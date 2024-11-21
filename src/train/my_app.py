# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass, field
from typing import Any, List
from transformers import HfArgumentParser
from omegaconf import MISSING, OmegaConf  # Do not confuse with dataclass.MISSING
import hydra
from src.train.arguments import HydraConfig, DataArguments, ModelArguments, TrainingConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="hydra_config")
def my_app(cfg: HydraConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingConfig))
    data_args, model_args, training_args = parser.parse_dict(cfg.training_configs)
    print(data_args)
    print(model_args)
    print(training_args)

if __name__ == "__main__":
    my_app()