import itertools
import os

from pathlib import Path
from typing import Dict, Generator, Optional

from fanyi.imports import import_data
from fanyi.utils import sort_files


def load_data(
    source_directory: Optional[Path],
    limit: Optional[int],
) -> Generator[Dict[str, str], None, None]:
    """
    Lazy-load raw and translated text files, yielding dictionaries
    containing the raw text and translated text of each file.

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
        raise ValueError('source_directory cannot be None.')

    if not source_directory.exists():
        raise FileNotFoundError(f'Directory {source_directory} does not exist.')

    # Sort directories to ensure consistent ordering after importing
    raw_files = sort_files(source_directory.joinpath('raws'))
    translated_files = sort_files(source_directory.joinpath('translations'))

    if limit:
        raw_files = itertools.islice(raw_files, limit)
        translated_files = itertools.islice(translated_files, limit)

    for raw_path, translated_path in zip(raw_files, translated_files):
        # Import function only copies .txt files
        raw_text = raw_path.read_text()
        translated_text = translated_path.read_text()

        yield {
            'raw_text': raw_text,
            'translated_text': translated_text,
        }


if __name__ == '__main__':
    for data in load_data(
        import_data(Path(os.getcwd() + '/tests/custom_dataset/chinese/'), 'chinese'),
        limit=10,
    ):
        pass
