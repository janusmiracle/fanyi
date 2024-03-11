import argparse
from pathlib import Path

# Outlining the args, has not been tested


def main():
    parser = argparse.ArgumentParser(description='model.')
    parser.add_argument('n', '--name', type=str, help='Name of the files.')
    parser.add_argument(
        '-r',
        '--raw',
        type=Path,
        help='Path to the directory containing raw text files.',
    )
    parser.add_argument(
        '-t',
        '--translated',
        type=Path,
        help='Path to the directory containing translated text files.',
    )
    parser.add_argument(
        '-l', '--language', type=str, help='Language code (e.g., CN, JP, KR).'
    )
    parser.add_argument('--limit', type=int, help='Maximum number of files to load.')

    # Add an optional output directory for both the translated files + finetuned model
    args = parser.parse_args()

    name = args.name
    raw_directory = args.raw
    translated_directory = args.translated
    language = args.language
    limit = args.limit


if __name__ == '__main__':
    main()
