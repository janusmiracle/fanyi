import argparse

from pathlib import Path

from dugong.handler import Handler
from dugong.inference import Inference
from dugong.train import MarianTrainer

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

    # Training
    parser.add_argument(
        "--evaluation-strategy",
        type=str,
        default="steps",
        help="The evaluation strategy.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="The learning rate."
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=4,
        help="The batch size per GPU for training.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=4,
        help="The batch size per GPU for evaluation.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="The weight decay."
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=1,
        help="The maximum number of checkpoints to save.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=2,
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
        "--max-steps",
        type=int,
        default=400,
        help="The maximum number of training steps.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=100,
        help="The number of steps between logging information.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="The number of steps between checkpoint saves.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=10,
        help="The number of steps between evaluations.",
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

    # Hell yeah
    name = args.name
    train_file = args.train
    test_file = args.test
    source_lang = args.source
    target_lang = args.target
    evaluation_strategy = args.evaluation_strategy
    learning_rate = args.learning_rate
    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    weight_decay = args.weight_decay
    save_total_limit = args.save_total_limit
    num_train_epochs = args.num_train_epochs
    predict_with_generate = args.predict_with_generate
    fp16 = args.fp16
    logging_steps = args.logging_steps
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    translate_path = args.translate
    file_limit = args.file_limit

    # Run the actual functions
    handler = Handler(name, train_file, test_file)
    train_dir, test_dir = handler.import_files()
    main_dir = handler.source_dir()
    main_dir = main_dir / "translations"
    output_dir = handler.output_dir()

    marian_train = MarianTrainer(
        train_dir, test_dir, source_lang, target_lang, output_dir
    )
    training_args = marian_train.training_args(
        evaluation_strategy=evaluation_strategy,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        num_train_epochs=num_train_epochs,
        predict_with_generate=predict_with_generate,
        fp16=fp16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
    )
    model, tokenizer = marian_train.train_torch(training_args)

    inference = Inference(
        model,
        tokenizer,
        translate_path,
        main_dir,
        source_lang,
        target_lang,
        file_limit=file_limit,
    )

    inference.translate()


if __name__ == "__main__":
    main()
