import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset
import numpy as np

from ...tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_xlm_roberta import XLMRobertaTokenizer
from ..processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors
from ..processors.utils import InputFeatures, InputAugFeatures


logger = logging.getLogger(__name__)


@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(
        default='mnli',
        metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())}
    )
    data_dir: str = field(
        default='data/mnli',
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    # train_aug_file: Optional[str] = field(default='data/mnli/train_back_trans.tsv')
    # train_aug_file: Optional[str] = field(default='output/mnli_croberta_bs32acc2_lr1e-5_ep10/checkpoint-50000/train_cbert.tsv')
    train_aug_file: Optional[str] = field(default='data/mnli/train_roberta_base.tsv')
    dev_aug_file: Optional[str] = field(default=None)

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        evaluate=False,
    ):
        self.args = args
        processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                "dev" if evaluate else "train", tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")
                label_list = processor.get_labels()
                if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
                    RobertaTokenizer,
                    RobertaTokenizerFast,
                    XLMRobertaTokenizer,
                ):
                    # HACK(label indices are swapped in RoBERTa pretrained model)
                    label_list[1], label_list[2] = label_list[2], label_list[1]
                examples = (
                    processor.get_dev_examples(args.data_dir)
                    if evaluate
                    else processor.get_train_examples(args.data_dir)
                )
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


class GlueAugDataset(Dataset):
    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]
    aug_features: List[InputFeatures]

    def __init__(self, args: GlueDataTrainingArguments, tokenizer: PreTrainedTokenizer,
                 limit_length: Optional[int] = None, evaluate=False):
        self.args = args
        processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        aug_processor = glue_processors[args.task_name + '-aug']()
        cached_features_file = os.path.join(
            args.data_dir,
            'cached_{}_{}_{}_{}'.format(
                'dev' if evaluate else 'train', tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            )
        )
        if not evaluate:
            _, _aug_file = os.path.split(args.train_aug_file)
            _aug_file = _aug_file.split('.')[0]
            _aug_file = _aug_file.replace('_', '-')
        else:
            _, _aug_file = os.path.split(args.dev_aug_file)
            _aug_file = _aug_file.split('.')[0]
            _aug_file = _aug_file.replace('_', '-')
        cached_aug_features_file = os.path.join(
            args.data_dir,
            'cached_{}_{}_{}_{}'.format(
                _aug_file, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            )
        )

        lock_path = cached_features_file + '.lock'
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(f'Loading features from cached file {cached_features_file} [took %.3f s]', time.time() - start)
            else:
                logger.info(f'Creating features from dataset file at {args.data_dir}')
                label_list = processor.get_labels()
                if args.task_name in ['mnli', 'mnli-mm'] and tokenizer.__class__ in (
                    RobertaTokenizer, RobertaTokenizerFast, XLMRobertaTokenizer
                ):
                    label_list[1], label_list[2] = label_list[2], label_list[1]
                examples = (
                    processor.get_dev_examples(args.data_dir) if evaluate else
                    processor.get_train_examples(args.data_dir)
                )
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                logger.info('Saving features into cached file %s [took %.3f s]', cached_features_file, time.time() - start)

        lock_path = cached_aug_features_file + '.lock'
        with FileLock(lock_path):
            if os.path.exists(cached_aug_features_file) and not args.overwrite_cache:
                start = time.time()
                self.aug_features = torch.load(cached_aug_features_file)
                logger.info(f'Loading aug features from cached file {cached_aug_features_file} [took %.3f s]', time.time() - start)
            else:
                logger.info(f'Creating aug features from dataset file at {args.dev_aug_file if evaluate else args.train_aug_file}')
                label_list = processor.get_labels()
                if args.task_name in ['mnli', 'mnli-mm'] and tokenizer.__class__ in (
                        RobertaTokenizer, RobertaTokenizerFast, XLMRobertaTokenizer
                ):
                    label_list[1], label_list[2] = label_list[2], label_list[1]
                examples = (
                    aug_processor.get_dev_aug_examples(args.dev_aug_file, args.aug_type) if evaluate else
                    aug_processor.get_train_aug_examples(args.train_aug_file, args.aug_type)
                )
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.aug_features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.aug_features, cached_aug_features_file)
                logger.info('Saving aug features into cached file %s [took %.3f s]', cached_aug_features_file, time.time() - start)

        assert len(self.features) == len(self.aug_features)

        # if not parallel:
        #     logger.warning('')
        #     np.random.shuffle(self.aug_features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputAugFeatures:
        feature_i = self.features[i]
        aug_feature_i = self.aug_features[i]
        return InputAugFeatures(
            input_ids=feature_i.input_ids,
            aug_input_ids=aug_feature_i.input_ids,
            attention_mask=feature_i.attention_mask,
            token_type_ids=feature_i.token_type_ids,
            label=feature_i.label,
            aug_attention_mask=aug_feature_i.attention_mask,
            aug_token_type_ids=aug_feature_i.token_type_ids,
            aug_label=aug_feature_i.label,
        )
