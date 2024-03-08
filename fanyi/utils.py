import re

# import natsort
from pathlib import Path

# Utility functions


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


def sort_files(path: Path) -> None:
    """
    Sort text files in a directory using natural sorting.

    Parameters
    ----------
    path : Path
        Path leading to directory.
    """

    return
