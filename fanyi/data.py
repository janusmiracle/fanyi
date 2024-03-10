from pathlib import Path
from nltk.tokenize import sent_tokenize
from typing import Generator, Dict, List

from fanyi.utils import sort_files

# nltk.download('punkt')


def load_data(directory: Path | None) -> Generator[Dict[str, List[str]], None, None]:
    """
    Lazy load text files from a directory and yield a dictionary
    containing the filename and the raw text of each file.

    Parameters
    ----------
    directory : Path
        Path to the directory containing the text files.

    Yields
    -------
    Dict[str, List[str]]
        A dictionary containing the filename and the raw text of each file.
    """
    if directory is None or not directory.exists():
        raise FileNotFoundError("Directory does not exist.")

    # Sort directory to ensure consistent order after importing
    sort_files(directory)

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()

            sentences = sent_tokenize(text_data)
            for sentence in sentences:
                yield {file_path.name: sentence}
