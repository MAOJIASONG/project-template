from typing import Dict, Optional
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from train.arguments import ModelArguments


class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__()
        self.config = encoder.config
        self.hidden_size = self.config.hidden_size
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def encode_input(self, input):
        hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states, input['attention_mask'])
        return pooled_output

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last':
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    @classmethod
    def build(cls, model_args: ModelArguments, **hf_kwargs):
        # Loading the base model
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        config.use_cache = False
        config.padding_side = "right"
        base_model = cls.TRANSFORMER_CLS.from_pretrained(
            model_args.model_name, **hf_kwargs, config=config, 
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True)
        base_model.padding_side = "right"

        if model_args.lora:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, **hf_kwargs):
        # Loading the base model
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        config.use_cache = False
        config.padding_side = "right"

        checkpoint_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        base_model = cls.TRANSFORMER_CLS.from_pretrained(
            checkpoint_path, **hf_kwargs, config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)
        base_model.padding_side = "right"

        # Building the model on top of the base
        if model_args.lora:
            lora_config = LoraConfig.from_pretrained(checkpoint_path)
            lora_model = PeftModel.from_pretrained(base_model, checkpoint_path, config=lora_config)
            
            merged_model = lora_model.merge_and_unload()
            model = cls(
                encoder=merged_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize
            )
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None):
        qry_reps = self.encode_input(qry) if qry else None  # (bsz_per_device, dim)
        tgt_reps = self.encode_input(tgt) if tgt else None # (bsz_per_device, dim)

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size

        return loss

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))



from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import LlamaModel, LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING
from transformers.utils import ModelOutput
from transformers.utils import add_start_docstrings_to_model_forward


class GatingNetwork(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, temperature: float = 10,
                 logit_scale: float = 1., hidden_dim: int = 1024, n_hidden: int = 3):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale)
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Apply the linear layers with ReLU
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        # Apply the conditional ReLU using the expanded mask
        x = F.softmax(x / self.temperature, dim=1)
        return x * self.logit_scale[0]


# token_pattern = tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False, )
token_pattern = [128009, 128006, 78191, 128007, 271]


def find_token_for_gating(lst, ):
    """Find the last occurrence of a token_pattern in a list."""
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j:j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")


@dataclass
class CustomOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        hidden_state (`Tuple[torch.FloatTensor]` of length `config.num_hidden_layers`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        prompt_embedding (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            The embeddings of the prompt tokens.
        gating_output (`torch.FloatTensor` of shape `(batch_size, config.num_objectives)`):
            The logits for the gating network.
        score (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            The final reward score.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Same as score
    """

    rewards: torch.FloatTensor = None
    hidden_state: Optional[Tuple[torch.FloatTensor, ...]] = None
    prompt_embedding: Optional[torch.FloatTensor] = None
    gating_output: Optional[torch.FloatTensor] = None
    score: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class LlamaForRewardModelWithGating(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        config_dict = config.to_dict()
        self.num_objectives = config_dict.get("num_objectives", 19)
        self.regression_layer = nn.Linear(config.hidden_size, self.num_objectives, bias=False)
        self.post_init()
        # Not using torch.eye because it is not supported in BF16
        I = torch.zeros(self.num_objectives, self.num_objectives)
        I[range(self.num_objectives), range(self.num_objectives)] = 1.
        self.reward_transform_matrix = nn.Parameter(I)
        self.reward_transform_matrix.requires_grad = False

        # Initialize weights and apply final processing
        self.gating = GatingNetwork(config.hidden_size, config.num_objectives,
                                    temperature=config_dict.get("gating_temperature", 10),
                                    hidden_dim=config_dict.get("gating_hidden_dim", 1024),
                                    n_hidden=config_dict.get("gating_n_hidden", 3))

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> CustomOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        tokens_hidden_states = transformer_outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(tokens_hidden_states.device)
            else:
                sequence_lengths = -1

        dummy_iterator = torch.arange(batch_size, device=tokens_hidden_states.device)
        hidden_states = tokens_hidden_states[dummy_iterator, sequence_lengths]
        assert hidden_states.shape == (batch_size, self.config.hidden_size)
        rewards = self.regression_layer(hidden_states)

        gating_token_positions = [find_token_for_gating(ids.tolist()) for ids in input_ids]
        prompt_embedding = tokens_hidden_states[dummy_iterator, gating_token_positions, :]
        gating_output = self.gating(prompt_embedding)

        rewards_adjusted = rewards @ self.reward_transform_matrix
        score = torch.sum(gating_output * rewards_adjusted, dim=1)

        return CustomOutput(
            rewards=rewards,
            hidden_state=hidden_states,
            prompt_embedding=prompt_embedding,
            gating_output=gating_output,
            score=score,
            logits=score,
        )