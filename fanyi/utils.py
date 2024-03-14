import nltk
import re

from natsort import natsorted
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from pathlib import Path
from typing import List


# Allow for multiple references?
def bleu_score(hypothesis_file: Path, reference_file: Path) -> float:
    """
    Calculates the BLEU score between the machine-translated text in the hypothesis file
    and the human-translated text in the reference file.

    Parameters
    ----------
    hypothesis_file : Path
        Path to the file containing the machine-translated text.
    reference_file : Path
        Path to the file containing the human-translated text.

    Returns
    -------
    bleu_score : float
        The BLEU score between 0 and 1, indicating the similarity between the two texts.
    """
    hypothesis_text = hypothesis_file.read_text()
    reference_text = reference_file.read_text()

    hypothesis = word_tokenize(hypothesis_text.lower())
    reference = word_tokenize(reference_text.lower())

    bleu_score = corpus_bleu([reference], [hypothesis])

    return bleu_score  # type: ignore


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
    return re.sub(r'[^\w\s\-\(\)\[\]]', '', name)


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
