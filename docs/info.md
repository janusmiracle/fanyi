# Installation
Clone the repository and navigate to the project directory.
```
$ git clone git@github.com:yaeso/dugong.git
$ cd dugong
```
Install the dependencies and activate the virtual environment. If you already have Poetry installed, you can run the shell command beforehand. 
```
$ poetry install
$ poetry shell
```

# Usage
Dugong is a tool for fine-tuning MarianMT models and using them for inference on translation tasks. It offers two main scripts: `train` and `translate`.

## Dataset
To train a model, first, create your custom dataset in the following JSON format:

```json
{
  "corpus": [
    {
      "en": "English text",
      "zh": "Chinese text"
    }
  ]
}
```
Note that the dataset MUST be labelled `corpus`. Ensure that each term is matched properly in your dataset.

## Training
The command for finetuning and inference is `poetry run train`.
```
$ poetry run train --help

usage: train [-h] [--name NAME] [--train TRAIN] [--test TEST] [--source SOURCE] [--target TARGET] [--evaluation-strategy EVALUATION_STRATEGY]
             [--learning-rate LEARNING_RATE] [--per-device-train-batch-size PER_DEVICE_TRAIN_BATCH_SIZE]
             [--per-device-eval-batch-size PER_DEVICE_EVAL_BATCH_SIZE] [--weight-decay WEIGHT_DECAY] [--save-total-limit SAVE_TOTAL_LIMIT]
             [--num-train-epochs NUM_TRAIN_EPOCHS] [--predict-with-generate] [--fp16] [--max-steps MAX_STEPS] [--logging-steps LOGGING_STEPS]
             [--save-steps SAVE_STEPS] [--eval-steps EVAL_STEPS] [--translate TRANSLATE] [--file-limit FILE_LIMIT]

Finetune MT/T5 models and use them for inference.

options:
  -h, --help            show this help message and exit
  --name NAME           Name for the output folder within 'data'.
  --train TRAIN         Path to JSON training data file.
  --test TEST           Path to JSON testing data file.
  --source SOURCE       Source language code (e.g. English = 'en').
  --target TARGET       Target language code (e.g. Chinese = 'zh').
  --evaluation-strategy EVALUATION_STRATEGY
                        The evaluation strategy.
  --learning-rate LEARNING_RATE
                        The learning rate.
  --per-device-train-batch-size PER_DEVICE_TRAIN_BATCH_SIZE
                        The batch size per GPU for training.
  --per-device-eval-batch-size PER_DEVICE_EVAL_BATCH_SIZE
                        The batch size per GPU for evaluation.
  --weight-decay WEIGHT_DECAY
                        The weight decay.
  --save-total-limit SAVE_TOTAL_LIMIT
                        The maximum number of checkpoints to save.
  --num-train-epochs NUM_TRAIN_EPOCHS
                        The total number of training epochs.
  --predict-with-generate
                        Whether to use generation during evaluation.
  --fp16                Whether to use 16-bit floating-point precision for training.
  --max-steps MAX_STEPS
                        The maximum number of training steps.
  --logging-steps LOGGING_STEPS
                        The number of steps between logging information.
  --save-steps SAVE_STEPS
                        The number of steps between checkpoint saves.
  --eval-steps EVAL_STEPS
                        The number of steps between evaluations.
  --translate TRANSLATE
                        Path leading to file or directory to translate after training.
  --file-limit FILE_LIMIT
                        Maximum number of files to load (only if a directory is passed to --translate).
```
Dugong offers a few training parameters from HuggingFace Trainer. If there are any others you wish to use, submit an Issue/PR or add them manually by editing `train.py` and `cli.py`.

An example of training and using them for inference inference without any training parameters set:

```
$ poetry run train --name Book --train dugong/examples/train_small.json --test dugong/examples/test_small.json --source zh --target en --translate tests/custom_dataset/chinese/raws/ --file-limit 1

/Users/user/dugong/data/Book created.

Files successfully imported to /Users/user/dugong/data/Book.

Generating train split: 0 examples [00:00, ? examples/s]
Generating train split: 1 examples [00:00, 604.98 examples/s]

Generating test split: 0 examples [00:00, ? examples/s]
Generating test split: 1 examples [00:00, 1492.63 examples/s]

Dataset loaded successfully!
╭──────────────────────────────────────────────────────────────────────── Dataset Preview ────────────────────────────────────────────────────────────────────────╮
│                                                                 Preview of the Training Dataset                                                                 │
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ │
│ ┃                        Source Sentence                         ┃                                      Target Sentence                                       ┃ │
│ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩ │
│ │              我要給阿Ｑ做正傳﹐已經不止一兩年了。              │         For several years now I have been meaning to write the true story of Ah Q.         │ │
│ │ 但一面要做﹐一面又往回想﹐這足見我不是一個〈立言〉〔２〕的人。 │  But while wanting to write I was in some trepidation, too, which goes to show that I am   │ │
│ │                                                                │                       not one of those who achieve glory by writing.                       │ │
│ │    因為從來不朽之筆﹐須傳不朽之人﹐於是人以文傳﹐文以人傳。    │  For an immortal pen has always been required to record the deeds of an immortal man, the  │ │
│ │                                                                │   man becoming known to posterity through the writing and the writing known to posterity   │ │
│ │                                                                │                                      through the man.                                      │ │
│ │               究竟誰靠誰傳﹐漸漸的不甚瞭然起來。               │                  Until finally it is not clear who is making whom known.                   │ │
│ └────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────┘ │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Tokenization complete!


Training... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% -:--:--
Training complete!
File 1 translation complete. Translation stored as translation-1.txt in /Users/user/dugong/data/Book/translations.
 
```
Model checkpoints and translations are outputted to `dugong/data/<name>`.

## Translations
For translation, use the `poetry run translate` script. Provide a file or directory of files to be translated:
```
$ poetry run translate --help   
usage: translate [-h] [--files FILES] [--name NAME] [--source SOURCE] [--target TARGET] [--file-limit FILE_LIMIT]

Translate inputted text files.

options:
  -h, --help            show this help message and exit
  --files FILES         Path leading to input file or directory of files.
  --name NAME           Name for the output folder within 'data'.
  --source SOURCE       Source language code (e.g. English = 'en').
  --target TARGET       Target language code (e.g. Chinese = 'zh').
  --file-limit FILE_LIMIT
                        Maximum number of files to load (only if a directory is passed to --translate). 
```
For example:
```
$ poetry run translate --files bookdirectory/ --name MyBook --source en --target zh --file-limit 5
```

