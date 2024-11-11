import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from arguments import DataArguments
import torch


logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
    data_args: DataArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        qry_inputs = self._get_batch_inputs(examples, 0, 1)
        pos_inputs = self._get_batch_inputs(examples, 2, 3)
        return qry_inputs, pos_inputs

    def _get_batch_inputs(self, examples, text_idx, image_idx):
        input_ids, pixel_values, image_sizes = [], [], []
        image_exist = False
        for example in examples:
            text, image = example[text_idx], example[image_idx]
            if image is None:
                inputs = self.processor(text, None, return_tensors="pt", max_length=self.data_args.max_len,
                                        truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
            else:
                inputs = self.processor(text, [image], return_tensors="pt", max_length=self.data_args.max_len, truncation=True)
                image_exist = True
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                pixel_values.append(inputs['pixel_values'])
                image_sizes.append(inputs['image_sizes'])

        input_ids = torch._C._nn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        ).squeeze(2)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        if not image_exist:
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        else:
            pixel_values = torch.cat(pixel_values, dim=0)
            image_sizes = torch.cat(image_sizes, dim=0)
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'image_sizes': image_sizes,
            }

        return inputs


@dataclass
class EvalCollator:
    data_args: DataArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, image_sizes = [], [], []
        image_exist = False
        for example in examples:
            text, image = example
            if image is None:
                inputs = self.processor(text, None, return_tensors="pt", max_length=self.data_args.max_len,
                                        truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
            else:
                inputs = self.processor(text, [image], return_tensors="pt", max_length=self.data_args.max_len, truncation=True)
                image_exist = True
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                pixel_values.append(inputs['pixel_values'])
                image_sizes.append(inputs['image_sizes'])

        input_ids = torch._C._nn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        ).squeeze(2)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        if not image_exist:
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        else:
            pixel_values = torch.cat(pixel_values, dim=0)
            image_sizes = torch.cat(image_sizes, dim=0)
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'image_sizes': image_sizes,
            }

        return inputs