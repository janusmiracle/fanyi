import evaluate
import torch
import warnings

from datasets import Dataset
from pathlib import Path
from rich.progress import Progress
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import (
    DataCollatorForSeq2Seq,
    logging,
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from typing import Optional, Tuple

from dugong.evaluation import Evaluate
from dugong.handler import Handler
from dugong.preprocess import Preprocessor

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

METRIC = evaluate.load("sacrebleu")


class BatchEncodingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.encodings.input_ids[idx])
        attention_mask = torch.tensor(self.encodings.attention_mask[idx])
        labels = torch.tensor(self.encodings.labels[idx])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self):
        return len(self.encodings.input_ids)


class MarianTrainer:
    """Train MarianMT Models."""

    def __init__(
        self,
        train_path: Path,
        test_path: Path,
        source_lang: str,
        target_lang: str,
        output_dir: Path,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.output_dir = output_dir

    def _setup(self) -> Tuple[
        Dataset,
        Dataset,
        MarianTokenizer,
        str,
        Evaluate,
        DataCollatorForSeq2Seq,
        MarianMTModel,
    ]:
        """Setup for training."""
        preprocessor = Preprocessor(
            source_lang=self.source_lang,
            target_lang=self.target_lang,
        )
        train_dataset, test_dataset = preprocessor.preprocess(
            self.train_path, self.test_path
        )
        tokenizer = preprocessor.get_tokenizer()
        checkpoint = preprocessor.get_checkpoint()

        evaluate = Evaluate(tokenizer, METRIC)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=checkpoint,
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,
        )

        model = MarianMTModel.from_pretrained(checkpoint)
        train_dataset = BatchEncodingDataset(train_dataset)
        test_dataset = BatchEncodingDataset(test_dataset)

        return (
            train_dataset,
            test_dataset,
            tokenizer,
            checkpoint,
            evaluate,
            data_collator,
            model,
        )

    def _training_args(self, **kwargs):
        """TODO: Handle training args here."""
        return

    def train_torch(self) -> MarianMTModel:
        """Trains MarianMT model with PyTorch."""
        (
            train_dataset,
            test_dataset,
            tokenizer,
            checkpoint,
            evaluate,
            data_collator,
            model,
        ) = self._setup()

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="steps",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            # weight_decay=0.01,
            # save_total_limit=3,
            # num_train_epochs=2,
            predict_with_generate=True,
            fp16=False,
            logging_steps=1,
            save_steps=1,
            eval_steps=1,
            max_steps=1,
        )
        with Progress() as progress:
            task = progress.add_task("[cyan]Training...", total=training_args.max_steps)
            print("\n")
            console = Console()

            for epoch in range(training_args.max_steps):
                trainer = Seq2SeqTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=lambda eval_preds: evaluate.compute_metrics(
                        eval_preds
                    ),
                )

                trainer.train()
                progress.update(task, advance=1)
                metrics = trainer.evaluate()

                metrics_table = Table(
                    title="Metrics", show_header=True, header_style="bold magenta"
                )
                metrics_table.add_column("Metric", justify="center")
                metrics_table.add_column("Value", justify="center")

                for metric_name, metric_value in metrics.items():
                    metrics_table.add_row(metric_name, str(metric_value))

                metrics_panel = Panel(
                    metrics_table, title="Evaluation Metrics", expand=False
                )
                console.print(metrics_panel)

        console.print("[green]Training complete![/green]")

        return model, tokenizer


class T5Trainer:
    """Train T5 Models."""

    def __init__(
        self,
        train_path: Path,
        test_path: Path,
        source_lang: str,
        target_lang: str,
        output_dir: Path,
        size: Optional[str] = "small",
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.output_dir = output_dir
        self.size = size

    def _setup(self) -> Tuple[
        Dataset,
        Dataset,
        T5Tokenizer,
        str,
        Evaluate,
        DataCollatorForSeq2Seq,
        T5ForConditionalGeneration,
    ]:
        """Setup for training."""
        preprocessor = Preprocessor(
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            model="t5",
            size=self.size,
        )
        train_dataset, test_dataset = preprocessor.preprocess(
            self.train_path, self.test_path
        )

        tokenizer = preprocessor.get_tokenizer()
        checkpoint = preprocessor.get_checkpoint()
        evaluate = Evaluate(tokenizer, METRIC)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=checkpoint,
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,
        )

        model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        train_dataset = BatchEncodingDataset(train_dataset)
        test_dataset = BatchEncodingDataset(test_dataset)

        return (
            train_dataset,
            test_dataset,
            tokenizer,
            checkpoint,
            evaluate,
            data_collator,
            model,
        )

    def _training_args(self):
        """TODO: handle T5 training args."""
        return

    def train_torch(self) -> T5ForConditionalGeneration:
        """Trains T5 model using PyTorch."""
        (
            train_dataset,
            test_dataset,
            tokenizer,
            checkpoint,
            evaluate,
            data_collator,
            model,
        ) = self._setup()

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=2,
            predict_with_generate=True,
            fp16=False,
        )

        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Training...", total=training_args.num_train_epochs
            )
            print("\n")
            console = Console()

            for epoch in range(training_args.num_train_epochs):
                trainer = Seq2SeqTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=lambda eval_preds: evaluate.compute_metrics(
                        eval_preds
                    ),
                )

                trainer.train()
                progress.update(task, advance=1)
                metrics = trainer.evaluate()

                metrics_table = Table(
                    title="Metrics", show_header=True, header_style="bold magenta"
                )
                metrics_table.add_column("Metric", justify="center")
                metrics_table.add_column("Value", justify="center")

                for metric_name, metric_value in metrics.items():
                    metrics_table.add_row(metric_name, str(metric_value))

                metrics_panel = Panel(
                    metrics_table, title="Evaluation Metrics", expand=False
                )
                console.print(metrics_panel)

        console.print("[green]Training complete![/green]")

        return model, tokenizer


if __name__ == "__main__":
    handler = Handler(
        "yippee",
        Path("dugong/examples/corpus_train.json"),
        Path("dugong/examples/corpus_test.json"),
    )
    train_dir, test_dir = handler.import_files()
    output_dir = handler.output_dir()

    marian_train = MarianTrainer(train_dir, test_dir, "zh", "en", output_dir)
    model = marian_train.train_torch()

    # translator = pipeline()
