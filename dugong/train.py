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
)
from typing import Tuple

from dugong.evaluation import Evaluate
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

    def training_args(
        self,
        evaluation_strategy: str = "steps",
        learning_rate: float = 5e-5,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        weight_decay: float = 0.0,
        save_total_limit: int = 5,
        num_train_epochs: int = 3,
        predict_with_generate: bool = False,
        fp16: bool = False,
        max_steps: int = -1,
        logging_steps: int = 500,
        save_steps: int = 500,
        eval_steps: int = None,
    ) -> Seq2SeqTrainingArguments:
        """Seq2SeqTrainingArguments for MarianMT."""
        return Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy=evaluation_strategy,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            weight_decay=weight_decay,
            save_total_limit=save_total_limit,
            num_train_epochs=num_train_epochs,
            predict_with_generate=predict_with_generate,
            fp16=fp16,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
        )

    def train_torch(self, training_args: Seq2SeqTrainingArguments) -> MarianMTModel:
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
