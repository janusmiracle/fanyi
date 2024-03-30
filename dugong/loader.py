import itertools
import os

from pathlib import Path
from typing import Dict, Generator, Optional

from dugong.imports import import_data
from dugong.utils import sort_files


def load_data(
    source_directory: Optional[Path],
    max_files: Optional[int],
) -> Generator[Dict[str, str], None, None]:
    """
    Lazy-load raw and translated text files, yielding dictionaries
    containing the raw text and translated text of each file.

    Parameters
    ----------
    source_directory : Optional[Path]
        Path to the directory containing 'raws' and 'translations' subdirectories,
        each containing .txt files to be loaded. If importing fails, None.
    max_files : Optional[int]
        Maximum number of files to load, by default None (load all files).

    Yields
    -------
    Dict[str, str]
         A dictionary containing the raw and translated text of each file.

    Raises
    ------
    ValueError
        If source_directory is None.
    FileNotFoundError
        If source_directory does not exist.
    """
    if source_directory is None:
        raise ValueError("Import has failed. source_directory cannot be None.")

    if not source_directory.exists():
        raise FileNotFoundError(f"Directory {source_directory} does not exist.")

    # Sort directories to ensure consistent ordering after importing
    raw_files = sort_files(source_directory.joinpath("raws"))
    translated_files = sort_files(source_directory.joinpath("translations"))

    if max_files:
        raw_files = itertools.islice(raw_files, max_files)
        translated_files = itertools.islice(translated_files, max_files)

    for raw_path, translated_path in zip(raw_files, translated_files):
        # Import function only copies .txt files
        raw_text = raw_path.read_text()
        translated_text = translated_path.read_text()

        # Load filenames for now, may change later (edit docstring whenever this is decided on)
        yield {
            "raw_filename": raw_path.name,
            "raw_text": raw_text,
            "translated_filename": translated_path.name,
            "translated_text": translated_text,
        }


if __name__ == "__main__":
    for data in load_data(
        import_data(Path(os.getcwd() + "/tests/custom_dataset/chinese/"), "chinese"),
        limit=10,
    ):
        pass
