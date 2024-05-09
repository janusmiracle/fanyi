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

from dugong.utils import load_files, load_sentences


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
        file_limit: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.file_limit = file_limit

    def _translator(self) -> TranslationPipeline:
        """Initialize pipeline translator."""
        translator = pipeline(
            task="translation", model=self.model, tokenizer=self.tokenizer
        )

        return translator

    def translate(self):
        """Translates text from a file. If given a directory, each file is translated."""
        translator = self._translator()
        translations = []
        count = 1
        for file in load_files(self.source_dir, self.file_limit):
            for sentence in load_sentences(file):
                translated_sentence = translator(sentence)
                translated_sentence = translated_sentence[0]["translation_text"]
                translations.append(translated_sentence + "\n")

            output_filename = f"translation-{count}.txt"
            output_path = self.output_dir.joinpath(output_filename)
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write("\n".join(translations))

                print(
                    f"File {count} translation complete. Translation stored as {output_filename} in {self.output_dir}.\n"
                )

                translations = []
                count += 1
