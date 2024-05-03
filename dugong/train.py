import evaluate
import torch
import warnings

from datasets import Dataset
from pathlib import Path
from rich.progress import Progress
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    MarianTokenizer,
    MarianMTModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    pipeline,
    logging,
)
from typing import Optional, Tuple

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

def MarianTrainer:
    """Train MarianMT Models."""
def setup(
    train_path: Path,
    test_path: Path,
    source_lang: str,
    target_lang: str,
    size: Optional[str] = "small",
) -> Tuple[
    Dataset,
    Dataset,
    MarianTokenizer,
    Evaluate,
    str,
    DataCollatorForSeq2Seq,
    MarianMTModel,
]:
    """Setup for training."""
    preprocessor = Preprocessor(
        model="helsinki", size=size, source_lang=source_lang, target_lang=target_lang
    )
    train_dataset, test_dataset = preprocessor.preprocess(train_path, test_path)
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


def train_torch(
    train_path: Path,
    test_path: Path,
    source_lang: str,
    target_lang: str,
    size: Optional[str] = "small",
):
    """Trains model using PyTorch."""
    (
        train_dataset,
        test_dataset,
        tokenizer,
        checkpoint,
        evaluate,
        data_collator,
        model,
    ) = setup(train_path, test_path, source_lang, target_lang, size)

    training_args = Seq2SeqTrainingArguments(
        output_dir="dugong/models/",
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
        max_steps=2,
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
                compute_metrics=lambda eval_preds: evaluate.compute_metrics(eval_preds),
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


if __name__ == "__main__":
    train_torch(
        Path("dugong/corpus_train.json"), Path("dugong/corpus_test.json"), "zh", "en"
    )

    translator = pipeline()
