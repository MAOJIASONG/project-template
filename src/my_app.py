# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass, field
from typing import Any, List
from transformers import HfArgumentParser
from omegaconf import MISSING, OmegaConf  # Do not confuse with dataclass.MISSING
import hydra
from src.train.arguments import DataArguments, ModelArguments, TrainingConfig
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../recipes", config_name="hydra_default")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # parser = HfArgumentParser((DataArguments, ModelArguments, TrainingConfig))
    # data_args, model_args, training_args = parser.parse_dict(cfg)
    # print(data_args)
    # print(model_args)
    # print(training_args)
    # with open("test.yaml", "w") as f:
    #     f.write(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
