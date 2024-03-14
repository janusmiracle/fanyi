import os
import re

from pathlib import Path
from typing import List, Optional, Tuple

from fanyi.errors import EmptyDirectoryError, InvalidCharacterError
from fanyi.paths import DATA_PATH
from fanyi.utils import clean_invalid


def import_validation(
    source_directory: Path,
    source_name: str,
    auto_clean: Optional[bool] = False,
) -> Tuple[List[str], str, Optional[Path]]:
    """
    Validates the input for importing raw text files or translations.

    Parameters
    ----------
    source_directory : Path
        Path to the directory containing text files to be imported.
    source_name : str
        Name for the output directory within the 'raws' or 'translations' directory.

    auto_clean : Optional[bool]
        Whether to automatically clean source_name if it contains illegal characters. (Defaults to False)

    Returns
    -------
    Tuple[List[str], str, Optional[Path]]
        A tuple containing a list of validation error messages, the cleaned source_name (if auto_clean is True, otherwise source_name), and the output directory (if there are no validation errors).
    """
    validation_errors = []

    try:
        validate_name(source_name)
    except InvalidCharacterError as e:
        if auto_clean:
            source_name = clean_invalid(source_name)
        else:
            validation_errors.append(f'InvalidCharacterError: {e}')

    try:
        validate_path(source_directory)
    except FileNotFoundError as e:
        validation_errors.append(f'FileNotFoundError: {e}')

    source_directory = DATA_PATH.joinpath(source_name)

    try:
        validate_output_directory(source_directory)
    except FileExistsError as e:
        validation_errors.append(f'FileExistsError: {e}')

    return validation_errors, source_name, source_directory


def validate_path(source_directory: Path) -> None:
    """
    Validates the the given path is a valid directory path.

    Raises
    ------
    FileNotFoundError
        If the source directory does not exist or if it is empty.
    EmptyDirectoryError
        If the source directory exists but is empty.
    """
    if not source_directory.is_dir():
        raise FileNotFoundError(
            f"Source directory '{source_directory}' does not exist."
        )

    if not os.listdir(source_directory):
        raise EmptyDirectoryError(source_directory)


def validate_output_directory(output_directory: Path) -> None:
    """
    Validates the output directory within the 'raws' directory.

    Raises
    ------
    FileExistsError
        If the output directory already exists.
    """
    if output_directory.exists():
        raise FileExistsError(f"Output directory '{output_directory}' already exists.")


def validate_name(source_name: str) -> None:
    """
    Validates output directory name for any illegal characters.

    Raises
    ------
    InvalidCharacterError
        If the inputted name contains an invalid character.
    """
    if re.search(r'[^a-zA-Z0-9_\-\(\)]', source_name):
        raise InvalidCharacterError(source_name)
