import os
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AdamW, BartTokenizer, get_linear_schedule_with_warmup
from transformers.modeling_bart import shift_tokens_right

from multitask_bart import BartForMultitaskLearning
from sampler import MultitaskSampler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class MultitaskBartFinetuner(pl.LightningModule):
    def __init__(self, hparams, get_dataset):
        super().__init__()
        self.hparams = hparams
        self.get_dataset = get_dataset

        self.model = BartForMultitaskLearning.from_pretrained(
            hparams.model_name_or_path
        )
        self.tokenizer = BartTokenizer.from_pretrained(
            hparams.tokenizer_name_or_path
        )

        self.tasks = self.hparams.tasks.split(",")

        # for loss weighting
        if hparams.loss_weights:
            self.loss_weights = [
                float(weight)
                for weight in self.hparams.loss_weights.split(",")
            ]
            assert len(self.tasks) == len(self.loss_weights)

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        lm_labels=None,
        use_cache=None,
        task=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
            use_cache=use_cache,
            task=task
        )

    def _step(self, batch):
        if batch["task"][0] == "response":
            pad_token_id = self.tokenizer.pad_token_id
            target_ids = batch["target_ids"]

            decoder_input_ids = shift_tokens_right(target_ids, pad_token_id)

            outputs = self(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                task=batch["task"][0]
            )

            lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                target_ids,
                self.hparams.label_smoothing,
                ignore_index=pad_token_id
            )

        elif batch["task"][0] in ["cfemotion", "emotion", "sentiment"]:
            outputs = self(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                lm_labels=batch["target_label"],
                task=batch["task"][0]
            )
            loss = outputs[0]

        else:
            raise ValueError("The dataset contains an invalid task.")

        if self.hparams.loss_weights:
            loss = self.loss_weights[self.tasks.index(batch["task"][0])] * loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {
            "avg_train_loss": avg_train_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs
        }

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {
            "avg_val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0
            }
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        using_native_amp=None
    ):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {
            "loss": "{:.3f}".format(self.trainer.avg_loss),
            "lr": self.lr_scheduler.get_last_lr()[-1]
        }
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = self.get_dataset(
            tokenizer=self.tokenizer,
            type_path="train",
            args=self.hparams
        )
        sampler = MultitaskSampler(
            data_source=train_dataset,
            batch_size=self.hparams.train_batch_size,
            drop_last=False
        )
        dataloader = DataLoader(
            dataset=train_dataset,
            batch_sampler=sampler,
            num_workers=4
        )
        t_total = (
            (
                len(dataloader.dataset)
                // (
                    self.hparams.train_batch_size * max(1, self.hparams.n_gpu)
                )
            )
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = self.get_dataset(
            tokenizer=self.tokenizer,
            type_path="val",
            args=self.hparams
        )
        sampler = MultitaskSampler(
            data_source=val_dataset,
            batch_size=self.hparams.train_batch_size,
            drop_last=False
        )
        return DataLoader(
            dataset=val_dataset,
            batch_sampler=sampler,
            num_workers=4
        )


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(
                pl_module.hparams.output_dir,
                "test_results.txt"
            )

            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write(
                            "{} = {}\n".format(key, str(metrics[key]))
                        )


args_dict = dict(
    data_dir="",  # path for data files
    output_dir="",  # path to save the checkpoints
    model_name_or_path="facebook/bart-large",
    tokenizer_name_or_path="facebook/bart-large",
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level="O1",  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
    label_smoothing=0.1
)
