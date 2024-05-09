from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer
from typing import Optional

from dugong.utils import load_files, load_sentences


def translate(
    files: Path,
    source_code: str,
    target_code: str,
    output_dir: Optional[Path] = Path("dugong/data/translations"),
    file_limit: Optional[int] = 1,
):
    """Translates input files."""
    model_name = f"Helsinki-NLP/opus-mt-{source_code.lower()}-{target_code.lower()}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    translations = []
    count = 1
    for file in load_files(files, file_limit):
        for sentence in load_sentences(file):
            input_ids = tokenizer.encode(
                sentence, return_tensors="pt", padding=True, truncation=True
            )
            tokenized_sentence = model.generate(input_ids, max_length=512, num_beams=4)
            translated_sentence = tokenizer.decode(
                tokenized_sentence[0], skip_special_tokens=True
            )
            if not translated_sentence == "_Other Organiser":
                translations.append(translated_sentence + "\n")

        output_filename = f"translation-{count}.txt"
        output_path = output_dir.joinpath(output_filename)
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write("\n".join(translations))

            print(
                f"File {count} translation complete. Translation stored as {output_filename} in {output_dir}.\n"
            )

            translations = []
            count += 1
