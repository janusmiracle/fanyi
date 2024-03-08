import os

from pathlib import Path
from typing import Generator, Dict
from fanyi.file_manager import import_raws

# Load data from 'raws' and 'translations'


def load_data(directory: Path | None) -> Generator[Dict[str, str], None, None]:
    """
    Lazy load text files from a directory and yield a dictionary
    containing the filename and the raw text of each file.

    Parameters
    ----------
    directory : Path
        Path to the directory containing the text files.

    Yields
    -------
    Dict[str, str]
        A dictionary containing the filename and the raw text of each file.
    """
    if directory is None or not directory.exists():
        raise FileNotFoundError("Directory does not exist.")

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()

            yield {file_path.name: text_data}


if __name__ == "__main__":
    directory = import_raws(Path(os.getcwd() + "/tests/test_files/raws/"), "test_raws")
    for file_data in load_data(directory):
        print(file_data, "\n")
