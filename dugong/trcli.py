import argparse

from pathlib import Path

from dugong.translate import translate


def main():

    parser = argparse.ArgumentParser(description="Translate inputted text files.")

    parser.add_argument(
        "--files", type=Path, help="Path leading to input file or directory of files."
    )

    parser.add_argument(
        "--name", type=str, help="Name for the output folder within 'data'."
    )
    parser.add_argument(
        "--source", type=str, help="Source language code (e.g. English = 'en')."
    )
    parser.add_argument(
        "--target", type=str, help="Target language code (e.g. Chinese = 'zh')."
    )
    parser.add_argument(
        "--file-limit",
        type=int,
        default=None,
        help="Maximum number of files to load (only if a directory is passed to --translate).",
    )

    args = parser.parse_args()

    files = args.files
    name = args.name
    source = args.source
    target = args.target
    file_limit = args.file_limit

    output_dir = Path("dugong/data").joinpath(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    translate(files, source, target, output_dir=output_dir, file_limit=file_limit)
