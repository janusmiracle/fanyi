# Fanyi

The goal of Fanyi is to provide better novel translations for avid readers. The process involves training translation models with raw novel chapters in the original language along with professionally human-translated chapters. The trained models are then used to provide higher-quality translations for untranslated chapters.

## Getting Started

Clone the repository:

```
$ git clone git@github.com:janusmiracle/fanyi.git
```

Install the required dependencies using Poetry:

```
$ cd fanyi
$ poetry install
```

## Usage

Prepare the files for training. Within the main import folder, create a subfolder named ````raws```` that contains the raw chapter text files, and another subfolder named ````translations```` that contains the corresponding human-translated text files.

E.g.

```
source/
├── raws/
│   ├── chapter1.txt
│   ├── chapter2.txt
│   └── ...
└── translations/
    ├── chapter1.txt
    ├── chapter2.txt
    └── ...
```

Fanyi will import and load the data from each file before training the model. It will then output a fine-tuned model.

Run the training script:

```
-s, --source: path leading to main folder containing raws & translation subfolders.
-n, --name: name of the novel/import folder within data.
-l, --language: language code of the raw files (determines what model to run)
                e.g. CN for Chinese, JP for Japanese, KR for Korean.
--limit: the number of files (from each) for the model to be trained on (OPTIONAL).
--output: the output path for the fine-tuned model (OPTIONAL).
````

Example:

```
$ poetry run train --source /path/source --name ARTOC --language KR --limit 10 --output /path/output
```

Use the trained model to generate translations for untranslated chapters:

```
-i, --input: input directory containing raw files to be translated.
-o, --output: output directory for the translated files.
--model: path leading to the model (OPTIONAL).
```

Example:

```
$ poetry run translate --input /path/input --output /path/output --model /path/to/model
```


# NOTE

The project is currently incomplete, the training and translation still needs to be done.

