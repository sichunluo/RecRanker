#!/usr/bin/env python3

import os
import random
import shutil
import logging
import transformers

import torch.distributed as dist

from datetime import datetime
from dataclasses import field, dataclass
from utils.util import set_logger, print_args

from utils.loader import PROCESSOR
from utils.trainer import LoggerCallback

from datasets import load_dataset
from transformers import (
    Trainer,
    set_seed,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
    default_data_collator
)

logger = logging.getLogger()

@dataclass
class DataArguments:
    no_load_model_pararmeters: bool = field(default=False)
    train_file: str = field(default=None)
    max_len: int = field(default=2048)
    processor: str = field(default='dialogue')
    preprocessing_num_workers: int = field(default=8)
    model_cfg: str = field(default="/data/models/llama")
    use_flash_attention: bool = field(default=False)
    stream: bool = field(default=False)

def resize(model, tokenizer, special_tokens):
    if len(special_tokens) == 0:
        return

    num_new_tokens = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train():
    parser = HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    timestr = datetime.now().strftime("-%Y-%m-%d-%H:%M")
    training_args.output_dir = training_args.output_dir + timestr
    training_args.logging_dir = os.path.join(training_args.output_dir, 'logging')

    if os.path.exists(training_args.output_dir):
        if training_args.overwrite_output_dir:
            if training_args.process_index == 0:
                shutil.rmtree(training_args.output_dir)
        else:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists. Use --overwrite_output_dir to overcome.")

    if training_args.world_size > 1:
        dist.barrier()

    if training_args.process_index == 0:
        os.makedirs(training_args.output_dir)

    if training_args.world_size > 1:
        dist.barrier()

    set_seed(training_args.seed)

    node_rank = int(os.getenv('GROUP_RANK', '0'))

    for _logger in [logger, transformers.utils.logging.get_logger(), logging.getLogger('DeepSpeed')]:
        set_logger(_logger, training_args.local_rank, data_args.stream,
                   os.path.join(training_args.output_dir, f'log-node-{node_rank}.log'))

    logger.warning("Device: %s, rank: %s, world size: %s", training_args.device, training_args.process_index,
                   training_args.world_size)

    if training_args.world_size > 1:
        dist.barrier()

    print_args(data_args, 'Data Arguments')
    print_args(training_args, 'Training Arguments')
    processor = PROCESSOR[data_args.processor]()

    if data_args.no_load_model_pararmeters:
        config = AutoConfig.from_pretrained(data_args.model_cfg)
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(data_args.model_cfg)
    tokenizer = AutoTokenizer.from_pretrained(data_args.model_cfg)

    if data_args.use_flash_attention:
        model = model.to_bettertransformer()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    resize(model, tokenizer, processor.get_special_token())
    train_sets = load_dataset('json', data_files=data_args.train_file.split(','), split='train')
    logger.info('Total %d case', len(train_sets))

    with training_args.main_process_first(desc="dataset map inputs"):
        train_sets = train_sets.map(
            processor.process_input,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running dialogue mapping on dataset",
        )

    with training_args.main_process_first(desc="dataset map tokenization"):
        train_sets = train_sets.map(
            processor.process_tokenize,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_len": data_args.max_len
            },
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on dataset",
        )

    with training_args.main_process_first(desc="Log a few random samples from the training set"):
        for index in random.sample(range(len(train_sets)), 3):
            label_tokens = []
            for label in train_sets[index]['labels']:
                if label >= 0:
                    label_tokens.append(tokenizer._tokenizer.id_to_token(label))
                else:
                    label_tokens.append('<|ignore|>')

            logger.info(
                "Sample %d of the raw training set:\n\ntext: %s\n\ninput_tokens: %s\n\nlabel_tokens: %s",
                index, train_sets[index]['text'],
                tokenizer.convert_ids_to_tokens(train_sets[index]['input_ids']),
                label_tokens
            )

    column_names = list(train_sets.features)
    with training_args.main_process_first(desc="dataset map grouping"):
        train_sets = train_sets.map(
            processor.group_texts,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_len": data_args.max_len
            },
            batched=True,
            remove_columns=column_names,
            num_proc=data_args.preprocessing_num_workers,
            desc=f"Grouping texts in chunks of {data_args.max_len}",
        )

    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_sets,
        data_collator=default_data_collator,
        callbacks=[LoggerCallback]
    )

    trainer.train()

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logging.exception(e)