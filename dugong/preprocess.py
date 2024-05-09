from datasets import Dataset, load_dataset
from pathlib import Path
from rich.progress import Progress
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import MarianTokenizer, T5Tokenizer
from typing import Tuple, Union

from dugong.handler import Handler


class Preprocessor:
    """
    Loads and preprocesses datasets for training.

    The preprocessor loads datasets from JSON and initializes a Dataset in Hugging Face format.

    A MarianMT or T5Tokenizer is also initialized.
    """

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.tokenizer, self.checkpoint = self._load_model()

    def get_tokenizer(self) -> Union[T5Tokenizer, MarianTokenizer]:
        """Returns the initialized T5 tokenizer."""
        return self.tokenizer

    def get_checkpoint(self) -> str:
        """Returns the checkpoint."""
        return self.checkpoint

    def _load_model(self) -> Tuple[Union[T5Tokenizer, MarianTokenizer], str]:
        """Loads either a MarianMT or T5 Model and initializes a tokenizer."""
        checkpoint = f"Helsinki-NLP/opus-mt-{self.source_lang.lower()}-{self.target_lang.lower()}"
        tokenizer = MarianTokenizer.from_pretrained(checkpoint)

        return tokenizer, checkpoint

    def _tokenize(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Tokenizes dataset and outputs model inputs."""
        tokenizer = self.tokenizer

        train_inputs = [
            example[self.source_lang] for example in dataset["train"]["corpus"][0]
        ]

        train_targets = [
            example[self.target_lang] for example in dataset["train"]["corpus"][0]
        ]
        train_model_inputs = tokenizer(
            train_inputs, text_target=train_targets, max_length=512, truncation=True
        )

        test_inputs = [
            example[self.source_lang] for example in dataset["test"]["corpus"][0]
        ]
        test_targets = [
            example[self.target_lang] for example in dataset["test"]["corpus"][0]
        ]
        test_model_inputs = tokenizer(
            test_inputs, text_target=test_targets, max_length=512, truncation=True
        )

        return train_model_inputs, test_model_inputs

    def _dataset(self, train_path: Path, test_path: Path) -> Dataset:
        """Load dataset from JSON and convert to Hugging Face Dataset format."""
        console = Console()

        with Progress(transient=True) as _:

            dataset = load_dataset(
                "json",
                data_files={
                    "train": str(train_path),
                    "test": str(test_path),
                },
            )
            if (
                "corpus" not in dataset["train"].column_names
                or "corpus" not in dataset["test"].column_names
            ):
                raise ValueError(
                    "Input train or test dataset formatted incorrectly, 'corpus' column not found."
                )

            train_dataset = dataset["train"]["corpus"]
            test_dataset = dataset["test"]["corpus"]

            if len(train_dataset) == 0:
                raise ValueError("Error loading dataset -- train dataset is empty.")
            if len(test_dataset) == 0:
                raise ValueError("Error loading dataset -- test dataset is empty.")

            source_sentences = [
                entry[self.source_lang]
                for entry in train_dataset[0]
                if entry[self.source_lang] is not None
            ]
            target_sentences = [
                entry[self.target_lang]
                for entry in train_dataset[0]
                if entry[self.target_lang] is not None
            ]

            if len(source_sentences) != len(target_sentences):
                raise ValueError(
                    f"{self.source_lang} and {self.target_lang} sentences do not match in length in training set."
                )

            source_sentences = [
                entry[self.source_lang]
                for entry in test_dataset[0]
                if entry[self.source_lang] is not None
            ]
            target_sentences = [
                entry[self.target_lang]
                for entry in test_dataset[0]
                if entry[self.target_lang] is not None
            ]

            if len(source_sentences) != len(target_sentences):
                raise ValueError(
                    f"{self.source_lang} and {self.target_lang} sentences do not match in length in testing set."
                )

            train_dataset = train_dataset[0]
            test_dataset = test_dataset[0]

            preview_table = Table(title="Preview of the Training Dataset")
            preview_table.add_column("Source Sentence", justify="center")
            preview_table.add_column("Target Sentence", justify="center")
            for i in range(min(5, len(train_dataset))):
                preview_table.add_row(
                    train_dataset[i][self.source_lang],
                    train_dataset[i][self.target_lang],
                )
            preview_panel = Panel(preview_table, title="Dataset Preview", expand=False)

            console.print("[green]Dataset loaded successfully![/green]")
            console.print(preview_panel)

        return dataset

    def preprocess(self, train_path: Path, test_path: Path) -> Tuple[Dataset, Dataset]:
        """Preprocesses the training and testing datasets."""
        dataset = self._dataset(train_path, test_path)

        train_tokenized_dataset, test_tokenized_dataset = self._tokenize(dataset)

        print("Tokenization complete!")
        return train_tokenized_dataset, test_tokenized_dataset
