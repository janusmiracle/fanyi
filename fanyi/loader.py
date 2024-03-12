import itertools
import os
from pathlib import Path
from typing import Dict, Generator, Optional

from fanyi.imports import import_data
from fanyi.utils import sort_files

# nltk.download('punkt')


def load_data(
    source_directory: Optional[Path],
    limit: Optional[int],
) -> Generator[Dict[str, str], None, None]:
    """
    Lazy load raw and translated text files from directories and yield a dictionary
    containing the filename and the raw and translated text of each file.

    Parameters
    ----------
    source_directory : Optional[Path]
        Path to the directory containing 'raws' and 'translations' subdirectories,
        each containing .txt files to be loaded. If importing fails, None.
    limit : Optional[int]
        Maximum number of files to load, by default None (load all files).

    Yields
    -------
    Dict[str, str]
         A dictionary containing the filename and the raw and translated text of each file.

    Raises
    ------
    ValueError
        If source_directory is None.
    FileNotFoundError
        If source_directory does not exist.
    """
    if source_directory is None:
        raise ValueError("source_directory cannot be None")

    if not source_directory.exists():
        raise FileNotFoundError(f"Directory {source_directory} does not exist.")

    # Sort directories to ensure consistent ordering after importing
    raw_files = sort_files(source_directory.joinpath("raws"))
    translated_files = sort_files(source_directory.joinpath("translations"))

    if limit:
        raw_files = itertools.islice(raw_files, limit)
        translated_files = itertools.islice(translated_files, limit)

    for raw_path, translated_path in zip(raw_files, translated_files):
        if (
            raw_path.is_file()
            and raw_path.suffix == ".txt"
            and translated_path.is_file()
            and translated_path.suffix == ".txt"
        ):
            with open(raw_path, "r", encoding="utf-8") as raw_file, open(
                translated_path, "r", encoding="utf-8"
            ) as translated_file:
                raw_text_data = raw_file.read()
                translated_text_data = translated_file.read()

            yield {
                "raw_filename": raw_path.name,
                "raw_text": raw_text_data,
                "translated_filename": translated_path.name,
                "translated_text": translated_text_data,
            }


if __name__ == "__main__":
    for data in load_data(
        import_data(Path(os.getcwd() + "/tests/custom_dataset/korean/"), "korean"),
        limit=10,
    ):
        print(data, "\n\n")
