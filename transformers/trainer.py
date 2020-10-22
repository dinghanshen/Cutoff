""" Cutoff: A Simple but Tough-to-Beat Data Augmentation Approach for Natural Language Understanding and Generation.  """

import json
import logging
import math
import os
import random
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange

from .data.data_collator import DataCollator, DefaultDataCollator
from .modeling_utils import PreTrainedModel
from .optimization import AdamW, get_linear_schedule_with_warmup
from .trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput
from .training_args import TrainingArguments, is_tpu_available
from utils import report_results

from .modeling_roberta import RobertaForMaskedLM, RobertaForSequenceClassification

try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False

def is_apex_available():
    return _has_apex

if is_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

def is_tensorboard_available():
    return False

try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except ImportError:
    _has_wandb = False

def is_wandb_available():
    return _has_wandb


logger = logging.getLogger()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

def get_tpu_sampler(dataset: Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())

def js_div(p, q):
    m = (p + q) / 2
    a = F.kl_div(p.log(), m, reduction='batchmean')
    b = F.kl_div(q.log(), m, reduction='batchmean')
    jsd = ((a + b) / 2)
    return jsd

def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
    ):
        """
        Trainer is a simple but feature-complete training and eval loop for PyTorch,
        optimized for Transformers.

        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.model = model.to(args.device)
        self.args = args
        if data_collator is not None:
            self.data_collator = data_collator
        else:
            self.data_collator = DefaultDataCollator()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        # if tb_writer is not None:
        #     self.tb_writer = tb_writer
        # elif is_tensorboard_available() and self.is_world_master():
        #     self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        self.tb_writer = None
        self.eval_history = []
        self.eval_header = None
        self.eval_key_axis = None
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        if is_wandb_available():
            self._setup_wandb()
        else:
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_world_master():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator.collate_batch,
        )

        if is_tpu_available():
            data_loader = pl.ParallelLoader(data_loader, [self.args.device]).per_device_loader(self.args.device)

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_tpu_available():
            sampler = SequentialDistributedSampler(
                eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(eval_dataset)
        else:
            sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )

        if is_tpu_available():
            data_loader = pl.ParallelLoader(data_loader, [self.args.device]).per_device_loader(self.args.device)

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        if is_tpu_available():
            sampler = SequentialDistributedSampler(
                test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(test_dataset)
        else:
            sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )

        if is_tpu_available():
            data_loader = pl.ParallelLoader(data_loader, [self.args.device]).per_device_loader(self.args.device)

        return data_loader

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        beta1, beta2 = self.args.adam_betas.split(',')
        beta1, beta2 = float(beta1), float(beta2)
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon,
                          betas=(beta1, beta2))
        if self.args.warmup_steps > 0:
            warmup_steps = self.args.warmup_steps
            self.args.warmup_ratio = warmup_steps / num_training_steps
        elif self.args.warmup_ratio > 0:
            warmup_steps = int(self.args.warmup_ratio * num_training_steps)
            self.args.warmup_steps = warmup_steps
        else:
            warmup_steps = 0
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:

        Environment:
            WANDB_WATCH:
                (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
                or "all" to log gradients and parameters
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        if self.args.do_train:
            logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
            _, run_name = os.path.split(self.args.output_dir)
            wandb.init(name=run_name, project="data_aug", config=vars(self.args))
            # keep track of model topology and gradients
            if os.getenv("WANDB_WATCH") != "false":
                wandb.watch(
                    self.model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.args.logging_steps)
                )

    def num_examples(self, dataloader: Union[DataLoader, "pl.PerDeviceLoader"]) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        if is_tpu_available():
            assert isinstance(dataloader, pl.PerDeviceLoader)
            return len(dataloader._loader._loader.dataset)
        else:
            return len(dataloader.dataset)

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )

        self.eval_history = []
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch-{epoch}", disable=not self.is_local_master())
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if self.args.do_aug:
                    if self.args.aug_type == 'span_cutoff':
                        step_loss = self._training_step_with_span_cutoff(model, inputs, optimizer)
                    elif self.args.aug_type == 'token_cutoff':
                        step_loss = self._training_step_with_token_cutoff(model, inputs, optimizer)
                    elif self.args.aug_type == 'dim_cutoff':
                        step_loss = self._training_step_with_dim_cutoff(model, inputs, optimizer)
                    else:
                        raise NotImplementedError
                else:
                    step_loss = self._training_step(model, inputs, optimizer)

                tr_loss += step_loss

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.max_grad_norm > 0:
                        if self.args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss

                        print()
                        self._log(logs)

                        # if self.args.evaluate_during_training and self.args.save_steps % self.args.logging_steps == 0:
                        #     self.evaluate()

                    if self.is_world_master() and self.args.evaluate_during_training and \
                            self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        self.evaluate_and_save_model(model, optimizer, scheduler)

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break

            if self.is_world_master() and self.args.evaluate_during_training:
                self.evaluate_and_save_model(model, optimizer, scheduler)

            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("\n\nTraining completed.\n\n")

        self.eval_history = sorted(self.eval_history, key=lambda x: x[0])
        for x in self.eval_history:
            del x[-1]
        report_results(self.eval_header, self.eval_history, axis=self.eval_key_axis)
        return TrainOutput(self.global_step, tr_loss / self.global_step)

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.tb_writer:
            for k, v in logs.items():
                self.tb_writer.add_scalar(k, v, self.global_step)
        if is_wandb_available() and self.args.do_train:
            wandb.log(logs, step=self.global_step)
        output = json.dumps({**logs, **{"step": self.global_step}})
        # if iterator is not None:
        #     iterator.write(output)
        # else:
        #     logger.info(output)
        logger.info(output)

    def _resolve_loss_item(self, loss, optimizer):
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def _training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        return self._resolve_loss_item(loss, optimizer)

    def generate_span_cutoff_embedding(self, embeds, masks, input_lens):
        input_embeds = []
        input_masks = []
        for i in range(embeds.shape[0]):
            cutoff_length = int(input_lens[i] * self.args.aug_cutoff_ratio)
            start = int(torch.rand(1) * (input_lens[i] - cutoff_length))
            cutoff_embed = torch.cat((embeds[i][:start],
                                      torch.zeros([cutoff_length, embeds.shape[-1]],
                                                  dtype=torch.float).to(self.args.device),
                                      embeds[i][start + cutoff_length:]), dim=0)
            cutoff_mask = torch.cat((masks[i][:start],
                                     torch.zeros([cutoff_length], dtype=torch.long).to(self.args.device),
                                     masks[i][start + cutoff_length:]), dim=0)
            input_embeds.append(cutoff_embed)
            input_masks.append(cutoff_mask)
        input_embeds = torch.stack(input_embeds, dim=0)
        input_masks = torch.stack(input_masks, dim=0)

        return input_embeds, input_masks

    def generate_token_cutoff_embedding(self, embeds, masks, input_lens):
        input_embeds = []
        input_masks = []
        for i in range(embeds.shape[0]):
            cutoff_length = int(input_lens[i] * self.args.aug_cutoff_ratio)
            zero_index = torch.randint(input_lens[i], (cutoff_length,))

            cutoff_embed = embeds[i]
            cutoff_mask = masks[i]

            tmp_mask = torch.ones(cutoff_embed.shape[0], ).to(self.args.device)
            for ind in zero_index:
                tmp_mask[ind] = 0

            cutoff_embed = torch.mul(tmp_mask[:, None], cutoff_embed)
            cutoff_mask = torch.mul(tmp_mask, cutoff_mask).type(torch.int64)

            input_embeds.append(cutoff_embed)
            input_masks.append(cutoff_mask)

        input_embeds = torch.stack(input_embeds, dim=0)
        input_masks = torch.stack(input_masks, dim=0)

        return input_embeds, input_masks

    def generate_dim_cutoff_embedding(self, embeds, masks, input_lens):
        input_embeds = []
        input_masks = []
        for i in range(embeds.shape[0]):
            cutoff_embed = embeds[i]
            cutoff_mask = masks[i]

            cutoff_length = int(cutoff_embed.shape[1] * self.args.aug_cutoff_ratio)
            zero_index = torch.randint(cutoff_embed.shape[1], (cutoff_length,))

            tmp_mask = torch.ones(cutoff_embed.shape[1], ).to(self.args.device)
            for ind in zero_index:
                tmp_mask[ind] = 0.

            cutoff_embed = torch.mul(tmp_mask, cutoff_embed)

            input_embeds.append(cutoff_embed)
            input_masks.append(cutoff_mask)
        input_embeds = torch.stack(input_embeds, dim=0)
        input_masks = torch.stack(input_masks, dim=0)

        return input_embeds, input_masks

    def _training_step_with_span_cutoff(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        ori_outputs = model(**inputs)
        loss = ori_outputs[0]  # model outputs are always tuple in transformers (see doc)

        assert model.__class__ is RobertaForSequenceClassification

        # Cut embedding_output and attention mask
        input_ids = inputs['input_ids']
        token_type_ids = inputs.get('token_type_ids', None)
        labels = inputs.get('labels', None)
        embeds = model.get_embedding_output(input_ids=input_ids, token_type_ids=token_type_ids)

        masks = inputs['attention_mask']
        input_lens = torch.sum(masks, dim=1)
        input_embeds = []
        input_masks = []
        for i in range(embeds.shape[0]):
            cutoff_length = int(input_lens[i] * self.args.aug_cutoff_ratio)
            start = int(torch.rand(1) * (input_lens[i] - cutoff_length))
            # print(input_lens[i], cutoff_length, start)
            cutoff_embed = torch.cat((embeds[i][:start],
                                      torch.zeros([cutoff_length, embeds.shape[-1]],
                                                  dtype=torch.float).to(self.args.device),
                                      embeds[i][start + cutoff_length:]), dim=0)
            cutoff_mask = torch.cat((masks[i][:start],
                                     torch.zeros([cutoff_length], dtype=torch.long).to(self.args.device),
                                     masks[i][start + cutoff_length:]), dim=0)
            input_embeds.append(cutoff_embed)
            input_masks.append(cutoff_mask)
        input_embeds = torch.stack(input_embeds, dim=0)
        input_masks = torch.stack(input_masks, dim=0)

        cutoff_outputs = model.get_logits_from_embedding_output(embedding_output=input_embeds,
                                                                attention_mask=input_masks, labels=labels)

        if self.args.aug_ce_loss > 0:
            loss += self.args.aug_ce_loss * cutoff_outputs[0]

        if self.args.aug_js_loss > 0:
            assert self.args.n_gpu == 1
            ori_logits = ori_outputs[1]
            aug_logits = cutoff_outputs[1]
            p = torch.softmax(ori_logits + 1e-10, dim=1)
            q = torch.softmax(aug_logits + 1e-10, dim=1)
            aug_js_loss = js_div(p, q)
            loss += self.args.aug_js_loss * aug_js_loss

        return self._resolve_loss_item(loss, optimizer)

    def _training_step_with_dim_cutoff(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        ori_outputs = model(**inputs)
        loss = ori_outputs[0]  # model outputs are always tuple in transformers (see doc)

        assert model.__class__ is RobertaForSequenceClassification
        input_ids = inputs['input_ids']
        token_type_ids = inputs.get('token_type_ids', None)
        labels = inputs.get('labels', None)
        embeds = model.get_embedding_output(input_ids=input_ids, token_type_ids=token_type_ids)

        masks = inputs['attention_mask']
        input_lens = torch.sum(masks, dim=1)

        input_embeds, input_masks = self.generate_dim_cutoff_embedding(embeds, masks, input_lens)
        cutoff_outputs = model.get_logits_from_embedding_output(embedding_output=input_embeds,
                                                               attention_mask=input_masks,
                                                               labels=labels)

        if self.args.aug_ce_loss > 0:
            loss += self.args.aug_ce_loss * cutoff_outputs[0]

        if self.args.aug_js_loss > 0:
            assert self.args.n_gpu == 1
            ori_logits = ori_outputs[1]
            aug_logits = cutoff_outputs[1]
            p = torch.softmax(ori_logits + 1e-10, dim=1)
            q = torch.softmax(aug_logits + 1e-10, dim=1)
            aug_js_loss = js_div(p, q)
            loss += self.args.aug_js_loss * aug_js_loss

        return self._resolve_loss_item(loss, optimizer)

    def _training_step_with_token_cutoff(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        ori_outputs = model(**inputs)
        #loss = ori_outputs[0]  # model outputs are always tuple in transformers (see doc)
        loss = 0.0

        assert model.__class__ is RobertaForSequenceClassification
        if self.args.aug_version == 'v3':
            input_ids = inputs['input_ids']
            token_type_ids = inputs.get('token_type_ids', None)
            labels = inputs.get('labels', None)
            embeds = model.get_embedding_output(input_ids=input_ids, token_type_ids=token_type_ids)

            masks = inputs['attention_mask']
            input_lens = torch.sum(masks, dim=1)

            input_embeds, input_masks = self.generate_token_cutoff_embedding(embeds, masks, input_lens)
            cutoff_outputs = model.get_logits_from_embedding_output(embedding_output=input_embeds,
                                                                   attention_mask=input_masks,
                                                                   labels=labels)

            if self.args.aug_ce_loss > 0:
                loss += self.args.aug_ce_loss * cutoff_outputs[0]

            if self.args.aug_js_loss > 0:
                assert self.args.n_gpu == 1
                ori_logits = ori_outputs[1]
                aug_logits = cutoff_outputs[1]
                p = torch.softmax(ori_logits + 1e-10, dim=1)
                q = torch.softmax(aug_logits + 1e-10, dim=1)
                aug_js_loss = js_div(p, q)
                loss += self.args.aug_js_loss * aug_js_loss

        return self._resolve_loss_item(loss, optimizer)

    def is_local_master(self) -> bool:
        if is_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if is_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Saving best-practices: if you use default names for the model,
        you can reload it using from_pretrained().

        Will only save from the world_master process (unless in TPUs).
        """

        if is_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_master():
            self._save(output_dir)

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        xm.rendezvous("saving_checkpoint")
        self.model.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _pop_checkpoints(self) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return
        key_metric = self.eval_header[self.eval_key_axis]
        if 'acc' in key_metric or 'f1' in key_metric:
            key = lambda x: x[self.eval_key_axis]
        elif 'loss' in key_metric or 'ppl' in key_metric:
            key = lambda x: -x[self.eval_key_axis]
        else:
            key = lambda x: x[self.eval_key_axis]
        self.eval_history = sorted(self.eval_history, key=key)
        if len(self.eval_history) <= self.args.save_total_limit:
            return
        number_of_checkpoints_to_delete = max(0, len(self.eval_history) - self.args.save_total_limit)
        checkpoints = self.eval_history[:number_of_checkpoints_to_delete]
        flag = False
        for checkpoint in checkpoints:
            step, checkpoint_path = checkpoint[0], checkpoint[-1]
            if os.path.exists(checkpoint_path):
                logger.info(f'Deleting checkpoint {checkpoint_path} ({key_metric} = {checkpoint[self.eval_key_axis]}) '
                            f'due to args.save_total_limit')
                shutil.rmtree(checkpoint_path)
                if step == self.global_step:
                    flag = True
        return flag

    def evaluate_and_save_model(self, model, optimizer, scheduler):
        metrics = self.evaluate()
        # In all cases (even distributed/parallel), self.model is always a reference
        # to the model we want to save.
        if hasattr(model, "module"):
            assert model.module is self.model
        else:
            assert model is self.model
        # Save model checkpoint
        output_dir = os.path.join(
            self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
        )

        if self.eval_header is None:
            self.eval_header = ['step'] + [metric for metric in sorted(metrics.keys()) if metric.startswith('eval_')]
            if 'eval_acc' in self.eval_header:
                self.eval_key_axis = self.eval_header.index('eval_acc')
            elif 'eval_mcc' in self.eval_header:
                self.eval_key_axis = self.eval_header.index('eval_mcc')
            #eval_pearson
            elif 'eval_pearson' in self.eval_header:
                self.eval_key_axis = self.eval_header.index('eval_pearson')
            elif 'eval_corr' in self.eval_header:
                self.eval_key_axis = self.eval_header.index('eval_corr')
            elif 'eval_acc_and_f1' in self.eval_header:
                self.eval_key_axis = self.eval_header.index('eval_acc_and_f1')
            elif 'eval_loss' in self.eval_header:
                self.eval_key_axis = self.eval_header.index('eval_loss')
            else:
                self.eval_key_axis = 1

        self.eval_history.append([self.global_step] + [metrics[k] for k in self.eval_header if k in metrics] + [output_dir])
        self.save_model(output_dir)
        if not self._pop_checkpoints():
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self._prediction_loop(eval_dataloader, description="Evaluation")

        self._log(output.metrics)

        if self.args.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output.metrics

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        test_dataloader = self.get_test_dataloader(test_dataset)
        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        if is_tpu_available():
            batch_size = dataloader._loader._loader.batch_size
        else:
            batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses = 0
        eval_size = 0
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        for inputs in tqdm(dataloader, desc=description, leave=False):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += step_eval_loss.mean().item() * logits.size()[0]
                    eval_size += logits.size()[0]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)
        print()
        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        elif is_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if eval_losses > 0:
            metrics["eval_loss"] = eval_losses / eval_size

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def distributed_concat(self, tensor: torch.Tensor, num_total_examples: int) -> torch.Tensor:
        assert self.args.local_rank != -1

        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)

        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        output = concat[:num_total_examples]
        return output