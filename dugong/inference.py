from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pathlib import Path
from transformers import (
    MarianTokenizer, MarianMTModel, pipeline,
    T5Tokenizer, T5ForConditionalGeneration
)
from typing import Union


class Inference:
    """Use finetuned models for inference."""

    def __init__(self, tokenizer: Union[MarianTokenizer, T5Tokenizer],
                 model: Union[MarianMTModel, T5ForConditionalGeneration]):
        self.model = model
        self.tokenizer = tokenizer

    def _translator(self):
        translator = pipeline(task="translation", model=self.model,
                              tokenizer=self.tokenizer)
        return translator

    def _score(self, references: Path, hypotheses: Optional[Path]):
        return

    def translate(self, references, hypotheses=Optional[]):
        """Translates reference text and provides a BLEU score if given a hypothesis."""
        return



reference = "The most annoying were some places on his scalp where in the past, at some uncertain date, shiny ringworm scars had appeared."
references = [[reference.split()]]

trained_model = "dugong/models/checkpoint-10"
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
translator = pipeline(task="translation", model=trained_model, tokenizer=tokenizer)

model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

text = "最惱人的是在他頭皮上﹐頗有幾處不知起於何時的癩瘡疤。"

base_output = model.generate(tokenizer.encode(text, return_tensors="pt"))
base_translation = tokenizer.decode(base_output[0], skip_special_tokens=True)

translated_text = [translator(text)[0]["translation_text"].split()]

base_hypotheses = [base_translation.split()]
base_bleu_score = corpus_bleu(references, base_hypotheses)

print(len(translated_text))
print(len(base_hypotheses))
smoother = SmoothingFunction().method4
base_bleu_score = corpus_bleu(references, base_hypotheses, smoothing_function=smoother)
trained_bleu_score = corpus_bleu(
    references, translated_text, smoothing_function=smoother
)

print(f"BLEU score for base MarianMT translation: {base_bleu_score * 100:.2f}")
print(f"BLEU score for trained model translation: {trained_bleu_score * 100:.2f}")
