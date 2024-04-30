from datasets import Dataset, load_dataset
from pathlib import Path
from rich.progress import Progress
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import MarianTokenizer, T5Tokenizer
from typing import Optional, Tuple, Union

from dugong.misc import LANGUAGE_CODES, T5_MODELS


class Preprocessor:
    """
    Loads and preprocesses datasets for training.

    The preprocessor loads datasets from JSON and
    initializes a Dataset in Hugging Face format.

    A MarianMT or T5Tokenizer is also initialized.

    Args:
        source_lang (str): The source language code.
        target_lang (str): The target language code.
        model (str, optional): The model to use ("t5" for T5 or MarianMT otherwise). Defaults to None.
        size (str, optional): The size of the model ("small", "base", "large"). Defaults to None.

    Attributes:
        tokenizer (Union[T5Tokenizer, MarianTokenizer]): The initialized tokenizer.
        checkpoint (str): The model checkpoint used.

    Methods:
        get_tokenizer: Returns the initialized tokenizer.
        get_checkpoint: Returns the model checkpoint.
        preprocess: Preprocesses the training and testing datasets.
    """

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        model: Optional[str] = None,
        size: Optional[str] = None,
    ):
        self.model = model
        self.size = size if model is not None and model.lower() == "t5" else "small"
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
        if self.model is not None and self.model.lower() == "t5":
            checkpoint = T5_MODELS[self.size.lower()]
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        else:
            checkpoint = f"Helsinki-NLP/opus-mt-{self.source_lang.lower()}-{self.target_lang.lower()}"
            tokenizer = MarianTokenizer.from_pretrained(checkpoint)

        return tokenizer, checkpoint

    def _prefix(self) -> str:
        """Set task prefix."""
        return f"translate {LANGUAGE_CODES[self.source_lang.lower()]} to {LANGUAGE_CODES[self.target_lang.lower()]}: "

    def _tokenize(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Tokenizes dataset and outputs model inputs."""
        tokenizer = self.tokenizer

        if self.model is not None and self.model.lower() == "t5":
            prefix = self._prefix()
        else:
            prefix = ""

        # + prefix if using T5
        train_inputs = [
            prefix + example[self.source_lang]
            for example in dataset["train"]["corpus"][0]
        ]

        train_targets = [
            example[self.target_lang] for example in dataset["train"]["corpus"][0]
        ]
        train_model_inputs = tokenizer(
            train_inputs, text_target=train_targets, max_length=512, truncation=True
        )

        test_inputs = [
            prefix + example[self.source_lang]
            for example in dataset["test"]["corpus"][0]
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

        with Progress(transient=True) as progress:
            task = progress.add_task("[cyan]Loading dataset...", total=4)

            dataset = load_dataset(
                "json",
                data_files={
                    "train": str(train_path),
                    "test": str(test_path),
                },
            )

            progress.update(task, advance=1)

            if (
                "corpus" not in dataset["train"].column_names
                or "corpus" not in dataset["test"].column_names
            ):
                raise ValueError(
                    "Input train or test dataset formatted incorrectly, 'corpus' column not found."
                )

            progress.update(task, advance=1)

            train_dataset = dataset["train"]["corpus"]
            test_dataset = dataset["test"]["corpus"]

            if len(train_dataset) == 0:
                raise ValueError("Error loading dataset -- train dataset is empty.")
            if len(test_dataset) == 0:
                raise ValueError("Error loading dataset -- test dataset is empty.")

            progress.update(task, advance=1)

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
                    f"{source_lang} and {target_lang} sentences do not match in length in training set."
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
                    f"{source_lang} and {target_lang} sentences do not match in length in testing set."
                )

            progress.update(task, advance=1)

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

            console.print("\n[green]Dataset loaded successfully![/green]\n")
            console.print(preview_panel)

        return dataset

    def preprocess(self, train_path: Path, test_path: Path) -> Tuple[Dataset, Dataset]:
        """Preprocesses the training and testing datasets."""
        dataset = self._dataset(train_path, test_path)

        train_tokenized_dataset, test_tokenized_dataset = self._tokenize(dataset)

        return train_tokenized_dataset, test_tokenized_dataset


if __name__ == "__main__":
    preprocessor = Preprocessor(source_lang="zh", target_lang="en")
    tokenizer = preprocessor.get_tokenizer()
    train_tokenized_dataset, test_tokenized_dataset = preprocessor.preprocess(
        Path("dugong/corpus_train.json"), Path("dugong/corpus_test.json")
    )
    # print(train_tokenized_dataset)

    decoded_dataset1 = tokenizer.batch_decode(
        test_tokenized_dataset["input_ids"], skip_special_tokens=True
    )
    decoded_dataset2 = tokenizer.batch_decode(
        test_tokenized_dataset["labels"], skip_special_tokens=True
    )

    print(len(train_tokenized_dataset["input_ids"]))
    print(len(train_tokenized_dataset["labels"]))
    # print("DECODED: \n")
    # print(decoded_dataset.keys())
