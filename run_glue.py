""" Cutoff: A Simple but Tough-to-Beat Data Augmentation Approach for Natural Language Understanding and Generation.  """
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional
import glob

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset, GlueAugDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

from utils import report_results


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.do_train and (os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        # filename=f'{training_args.output_dir}/log',
        # filemode='w',
    )
    # logger.addHandler(logging.StreamHandler(sys.stdout))
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(filename=f'{training_args.output_dir}/log', mode='w' if training_args.do_train else 'a'))
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    mnli_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=3,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    mnli_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=mnli_config,
        cache_dir=model_args.cache_dir,
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    pretrained_state_dict = {k: v for k, v in mnli_model.state_dict().items() if k != 'classifier.out_proj.bias' and k != 'classifier.out_proj.weight'}
    model.load_state_dict(pretrained_state_dict, strict=False)

    # Get datasets
    train_dataset_class = GlueDataset
    eval_dataset_class = GlueDataset
    if training_args.do_aug and training_args.aug_type:
        if training_args.aug_type in {'back_trans', 'cbert'}:
            if data_args.train_aug_file:
                train_dataset_class = GlueAugDataset
                data_args.aug_type = training_args.aug_type
            if data_args.dev_aug_file:
                eval_dataset_class = GlueAugDataset
                data_args.aug_type = training_args.aug_type
    train_dataset = train_dataset_class(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = eval_dataset_class(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval or \
                                                                                    training_args.do_eval_all else None

    def compute_metrics(p: EvalPrediction) -> Dict:
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

    if training_args.do_debug:
        eval_dataset = eval_dataset[:100]

    # training_args.do_aug = model_args.do_aug
    # training_args.aug_type = data_args.aug_type
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None,
        )
        # if not training_args.evaluate_during_training:
            # trainer.save_model()
            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    elif training_args.do_eval:
        # Evaluation
        results = {}
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, evaluate=True))

        for eval_dataset in eval_datasets:
            eval_output = trainer.evaluate(eval_dataset=eval_dataset)
            results[f'{eval_dataset.args.task_name}_acc'] = eval_output['eval_acc']
            results[f'{eval_dataset.args.task_name}_loss'] = eval_output['eval_loss']

        return results

    elif training_args.do_eval_all:
        results = []
        logger.info('*** Evaluate all checkpoints ***')

        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, evaluate=True))

        all_checkpoints = glob.glob(f'{training_args.output_dir}/checkpoint-*')
        all_checkpoints = sorted(all_checkpoints, key=lambda x: int(x.split('-')[-1]))

        for checkpoint in all_checkpoints:
            step = int(checkpoint.split('-')[-1])
            model.load_pretrained(checkpoint)
            model.to(training_args.device)
            step_result = [step]
            for eval_dataset in eval_datasets:
                trainer.global_step = step
                result = trainer.evaluate(eval_dataset=eval_dataset)
                # result['step'] = step
                step_result += [result['eval_acc'], result['eval_loss']]
            results.append(step_result)

        header = ['step']
        for eval_dataset in eval_datasets:
            header += [f'{eval_dataset.args.task_name}_acc', f'{eval_dataset.args.task_name}_loss']

        logger.info("***** Eval results *****")
        report_results(header, results, axis=1)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
