import argparse

from pathlib import Path
from typing import Optional

from fanyi.imports import import_data
from fanyi.loader import load_data


def import_and_load(
    source_directory: Path, source_name: str, max_files: Optional[int]
) -> None:
    output_directory = import_data(source_directory, source_name)

    if output_directory is None:
        # Unnecessary as this would be raised
        print('Import failed. Exiting.')
        return

    generator = load_data(output_directory, max_files)

    for index, data in enumerate(generator, start=1):
        print(f'\nFile {index}:')
        print(f"    - Raw Filename:         {data['raw_filename']}")
        print(f"    - Translated Filename:  {data['translated_filename']}")


def main():
    # Obviously, edit this later.
    parser = argparse.ArgumentParser(
        description='Import and load data from text files.'
    )
    parser.add_argument(
        '--source', '-s', type=Path, help='Path to the source directory.'
    )
    parser.add_argument('--name', '-n', type=str, help='Name for the output directory.')
    parser.add_argument('--max', type=int, help='Maximum number of files to load.')

    # parser.add_argument("-l", "--language", type=str, help="Language code (e.g., CN, JP, KR).")
    # Add an optional output directory for both the translated files + finetuned model
    args = parser.parse_args()

    import_and_load(args.source, args.name, args.max)


if __name__ == '__main__':
    main()
