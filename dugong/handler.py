import shutil
import os

from pathlib import Path
from typing import Optional, Tuple, Union

from dugong.paths import DATA_PATH

# from dugong.validations import Validation

VALID_EXTENSIONS = [".txt", ".json"]


class Handler:
    """File handler."""

    def __init__(self, name: str, train_file: Path, test_file: Optional[Path] = None):
        self.name = name
        self.train_file = train_file
        self.test_file = test_file

    def _suffix(self) -> Union[str, Tuple[str, str]]:
        """Checks file extension."""
        if self.test_file:
            train_suffix = os.path.splitext(self.train_file)[1].lower()
            test_suffix = os.path.splitext(self.test_file)[1].lower()

            if (
                train_suffix not in VALID_EXTENSIONS
                or test_suffix not in VALID_EXTENSIONS
            ):
                raise ValueError("Invalid file extension(s).")

            return train_suffix, test_suffix

        else:
            train_suffix = os.path.splitext(self.train_file)[1].lower()

            if train_suffix not in VALID_EXTENSIONS:
                raise ValueError("Invalid file extension.")

            return train_suffix

    def import_files(self):
        """Import training and, if included, test files."""
        self._suffix()

        output_directory = DATA_PATH.joinpath(self.name)
        output_directory.mkdir(parents=True, exist_ok=True)

        train_directory = output_directory.joinpath("train")
        train_directory.mkdir(exist_ok=True)
        shutil.copy(self.train_file, train_directory)

        if self.test_file:
            test_directory = output_directory.joinpath("test")
            test_directory.mkdir(exist_ok=True)
            shutil.copy(self.test_file, test_directory)

            return train_directory.joinpath(
                self.train_file.name
            ), test_directory.joinpath(self.test_file.name)

        return train_directory.joinpath(self.train_file.name), None


class Loader:
    """Lazy loads files from a directory and their text."""


if __name__ == "__main__":
    handler = Handler(
        "corpus",
        Path("dugong/examples/corpus_train.json"),
        Path("dugong/examples/corpus_test.json"),
    )
    output_directory = handler.import_files()
    print(output_directory)
