import itertools

from pathlib import Path
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    pipeline,
    T5Tokenizer,
    T5ForConditionalGeneration,
    TranslationPipeline,
)
from typing import Optional, Union

from dugong.handler import Handler
from dugong.train import MarianTrainer
from dugong.utils import sort_files


class Inference:
    """Use finetuned models for inference."""

    def __init__(
        self,
        model: Union[MarianMTModel, T5ForConditionalGeneration],
        tokenizer: Union[MarianTokenizer, T5Tokenizer],
        source_dir: Path,
        output_dir: Path,
        source_lang: str,
        target_lang: str,
        file_limit: Optional[int],
        t5: Optional[bool] = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.file_limit = file_limit
        self.t5 = t5

    def _translator(self) -> TranslationPipeline:
        if not self.t5:
            translator = pipeline(
                task="translation", model=self.model, tokenizer=self.tokenizer
            )

        else:
            translator = pipeline(
                f"task=translation_{self.source_lang}_to_{self.target_lang}",
                model=self.model,
                tokenizer=self.tokenizer,
            )

        return translator

    def load_data(self):
        """Lazy load files and their sentences."""
        files = sort_files(self.source_dir)

        # Only load up to file limit
        if self.file_limit:
            files = itertools.islice(files, self.file_limit)

        for idx, path in enumerate(files, start=1):
            with open(path, "r", encoding="utf-8") as file:
                sentences = file.read().split("\n")

                for sentence in sentences:
                    yield sentence

            yield None

    def translate(self):
        """Translates text from a file. If given a directory, each file is translated."""
        translator = self._translator()
        translations = []

        loader = self.load_data()
        count = 1
        try:
            while True:
                sentence = next(loader)
                if sentence is None:  # EOF
                    output_filename = f"translation-{count}.txt"
                    output_path = self.output_dir.joinpath(output_filename)
                    with open(output_path, "w", encoding="utf-8") as output_file:
                        output_file.write("\n".join(translations))

                    print(
                        f"File {count} translation complete. Translation stored as {output_filename} in {self.output_dir}.\n"
                    )
                    translations = []
                    count += 1
                else:
                    translated_sentence = translator(sentence)
                    if translated_sentence:
                        translated_sentence = translated_sentence[0]["translation_text"]
                        translations.append(translated_sentence)

        except StopIteration:
            pass


if __name__ == "__main__":
    handler = Handler(
        "yippee",
        Path("dugong/examples/corpus_train.json"),
        Path("dugong/examples/corpus_test.json"),
    )
    train_dir, test_dir = handler.import_files()
    main_dir = handler.source_dir()
    main_dir = main_dir / "translations"
    output_dir = handler.output_dir()
    checkpoint = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = MarianTokenizer.from_pretrained(checkpoint)
    marian_train = MarianTrainer(train_dir, test_dir, "zh", "en", output_dir)
    model, tokenizer = marian_train.train_torch()

    inference = Inference(
        model,
        tokenizer,
        Path("tests/custom_dataset/chinese/raws"),
        main_dir,
        "zh",
        "en",
        file_limit=1,
    )

    inference.translate()
