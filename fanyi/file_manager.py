import glob
import logging
import os
import shutil

from pathlib import Path
from typing import Optional
from .validation import import_validation
from .errors import ValidationError

# File handler for the program.


# Setup verbosity option that controls logger.setLevel
def import_raws(
    source_directory: Path,
    source_name: str,
    auto_clean: Optional[bool] = False,
) -> Optional[Path]:
    """
    Imports raw text files from a source directory into the 'raws' directory.

    Parameters
    ----------
    source_directory : Path
        Path to the directory containing text files to be imported.
    source_name : str
        Name for the output directory within the 'raws' directory.
    auto_clean : Optional[bool]
        Whether to automatically clean source_name if it contains illegal characters. (Defaults to False)
    """
    validation_errors, source_name, output_directory = import_validation(
        source_directory, source_name, str(import_raws.__name__), auto_clean=auto_clean
    )

    # Output errors
    if validation_errors:
        logging.error("Import validation has failed.\n")
        raise ValidationError("Validation failure: ", errors=validation_errors)

    logging.info("Import validation successful.\n")

    # Create subdirectory within 'raws'
    if output_directory is not None:
        os.mkdir(output_directory)

        # Populate output_directory with .txt files from source_directory
        for text_file in glob.glob(os.path.join(source_directory, "*.txt*")):
            shutil.copy(text_file, output_directory)

    logging.info(
        f"Success. '{source_name}' files have been imported to '{output_directory}'."
    )

    return output_directory


def import_translations(
    source_directory: Path,
    source_name: str,
    auto_clean: Optional[bool] = False,
) -> Optional[Path]:
    """
    Imports translations text files from a source directory into the 'translations' directory.

    Parameters
    ----------
    source_directory : Path
        Path to the directory containing text files to be imported.
    source_name : str
        Name for the output directory within the 'translations' directory.
    auto_clean : Optional[bool]
        Whether to automatically clean source_name if it contains illegal characters. (Defaults to False)

    Returns
    -------
    output_directory : Optional[Path]
        Path leading to the newly imported raw files. (None if validation fails)
    """
    validation_errors, source_name, output_directory = import_validation(
        source_directory,
        source_name,
        str(import_translations.__name__),
        auto_clean=auto_clean,
    )

    # Output errors
    if validation_errors:
        logging.error("Import validation has failed.\n")
        raise ValidationError("Validation failed.", errors=validation_errors)

    logging.info("Import validation successful.\n")

    # Create subdirectory within 'raws'
    if output_directory is not None:
        os.makedirs(output_directory)

        # Populate output_directory with .txt files from source_directory
        for text_file in glob.glob(os.path.join(source_directory, "*.txt*")):
            shutil.copy(text_file, output_directory)

    logging.info(
        f"Success. '{source_name}' files have been imported to '{output_directory}'."
    )

    return output_directory
