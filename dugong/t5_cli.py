import argparse

from pathlib import Path
from typing import Optional

from dugong.handler import Handler
from dugong.inference import Inference
from dugong.preprocess import Preprocessor
from dugong.train import MarianTrainer, T5Trainer

# from dugong.utils import check


def main():

    # Handler
    parser = argparse.ArgumentParser(
        description="Finetune MT/T5 models and use them for inference."
    )
    parser.add_argument(
        "--name", type=str, help="Name for the output folder within 'data'."
    )
    parser.add_argument("--train", type=Path, help="Path to JSON training data file.")
    parser.add_argument("--test", type=Path, help="Path to JSON testing data file.")

    # Preprocessor
    parser.add_argument(
        "--source", type=str, help="Source language code (e.g. English = 'en')."
    )
    parser.add_argument(
        "--target", type=str, help="Target language code (e.g. Chinese = 'zh')."
    )
    parser.add_argument(
        "--size",
        type=str,
        help="The T5 model size (small, base, large, 3b...).",
    )

    # Training
    parser.add_argument(
        "--evaluation-strategy",
        type=str,
        default="epochs",
        help="The evaluation strategy.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5, help="The learning rate."
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="The batch size per GPU for training.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8,
        help="The batch size per GPU for evaluation.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="The weight decay."
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=5,
        help="The maximum number of checkpoints to save.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="The total number of training epochs.",
    )
    parser.add_argument(
        "--predict-with-generate",
        action="store_true",
        help="Whether to use generation during evaluation.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit floating-point precision for training.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=500,
        help="The number of steps between logging information.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="The number of steps between checkpoint saves.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="The number of steps between evaluations (None for default behavior).",
    )

    # Inference
    parser.add_argument(
        "--translate",
        type=Path,
        help="Path leading to file or directory to translate after training.",
    )

    parser.add_argument(
        "--file-limit",
        type=int,
        default=None,
        help="Maximum number of files to load (only if a directory is passed to --translate).",
    )
    args = parser.parse_args()


if __name__ == "__main__":
    main()
