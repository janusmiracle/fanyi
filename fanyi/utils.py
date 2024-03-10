import re

from natsort import natsorted
from pathlib import Path
from typing import List


def bleu_score():
    return


def clean_invalid(name: str) -> str:
    """
    Removes invalid characters from a string.

    Parameters
    ----------
    name : str
        The input string to clean.

    Returns
    -------
    str
        The cleaned string with only valid characters.
    """
    return re.sub(r"[^\w\s\-\(\)\[\]]", "", name)


def sort_files(path: Path) -> List[Path]:
    """
    Sort text files in a directory using natural sorting.

    Parameters
    ----------
    path : Path
        Path leading to directory.
    """
    files = path.iterdir()
    sorted_files = natsorted(files)
    return list(sorted_files)
