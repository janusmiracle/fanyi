import os
import shutil

from pathlib import Path
from typing import Optional

from fanyi.errors import ValidationError
from fanyi.validation import import_validation


def import_data(
    source_directory: Path, source_name: str, auto_clean: Optional[bool] = False
) -> Optional[Path]:
    """
    Imports raw text files from a source directory into the 'raws' directory and
    corresponding translated text files into the 'translations' directory within
    a source-specific directory in the 'data' folder.

    Parameters
    ----------
    source_directory : Path
        Path to the directory containing text files to be imported.
    source_name : str
        Name for the output directory within the 'data' directory.
    auto_clean : Optional[bool]
        Whether to automatically clean source_name if it contains illegal characters.
        (Defaults to False)

    Returns
    -------
    output_directory : Optional[Path]
        Path leading to the newly imported raw and translated files.
        (None if validation fails)
    """
    validation_errors, source_name, output_directory = import_validation(
        source_directory,
        source_name,
        auto_clean=auto_clean,
    )

    # Output errors
    if validation_errors:
        raise ValidationError(
            'Import validation has failed.\n', errors=validation_errors
        )

    # Create source-specific directory within 'data'
    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)

        # Copy files to 'raws' and 'translations' directories within the source directory
        for directory_type in ['raws', 'translations']:
            directory_path = source_directory.joinpath(directory_type)
            output_directory_type = output_directory.joinpath(directory_type)
            os.makedirs(output_directory_type, exist_ok=True)

            for text_file in os.listdir(directory_path):
                if text_file.endswith('.txt'):
                    shutil.copy(
                        directory_path.joinpath(text_file), output_directory_type
                    )

    return output_directory


if __name__ == '__main__':
    output = import_data(Path(os.getcwd() + '/tests/custom_dataset/korean/'), 'ARTOC')
    print(output)
