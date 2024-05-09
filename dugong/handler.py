import shutil
import os

from pathlib import Path
from typing import Tuple

from dugong.misc import DATA_PATH


# from dugong.validations import Validation

VALID_EXTENSIONS = [".txt", ".json"]


class Handler:
    """Data handler for managing directories and importing files for a specified dataset."""

    def __init__(self, name: str, train_file: Path, test_file: Path):
        self.name = name
        self.train_file = train_file
        self.test_file = test_file

    def _suffix(self) -> Tuple[str, str]:
        """Checks for valid file extensions."""
        train_suffix = os.path.splitext(self.train_file)[1].lower()
        test_suffix = os.path.splitext(self.test_file)[1].lower()

        if train_suffix not in VALID_EXTENSIONS or test_suffix not in VALID_EXTENSIONS:
            raise ValueError("Invalid file extension(s).")

        return train_suffix, test_suffix

    def get_name(self) -> str:
        """Returns name."""
        return self.name

    def source_dir(self) -> Path:
        return DATA_PATH.joinpath(self.name)

    def output_dir(self) -> Path:
        return self.source_dir().joinpath("models")

    def create_folders(self) -> Path:
        """Creates train, test, models, and translations subdirectories."""
        source_directory = self.source_dir()
        source_directory.mkdir(parents=True, exist_ok=True)

        train_directory = source_directory.joinpath("train")
        train_directory.mkdir(exist_ok=True)

        test_directory = source_directory.joinpath("test")
        test_directory.mkdir(exist_ok=True)

        model_directory = source_directory.joinpath("models")
        model_directory.mkdir(exist_ok=True)

        translations_directory = source_directory.joinpath("translations")
        translations_directory.mkdir(exist_ok=True)

        print(f"{source_directory} created.")

        return source_directory

    def import_files(self) -> Tuple[Path, Path]:
        """Imports training and test files and returns their paths."""
        self._suffix()
        source_directory = self.create_folders()

        train_directory = source_directory.joinpath("train")
        test_directory = source_directory.joinpath("test")

        shutil.copy(self.train_file, train_directory)
        shutil.copy(self.test_file, test_directory)

        print(f"\nFiles successfully imported to {source_directory}.\n")

        return train_directory.joinpath(self.train_file.name), test_directory.joinpath(
            self.test_file.name
        )
