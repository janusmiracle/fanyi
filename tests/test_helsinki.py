import unittest

from transformers import MarianMTModel, MarianTokenizer

from fanyi.helsinki_models import HELSINKI_MODELS

# Add tests for large amounts of text.


class TestHelsinkiModels(unittest.TestCase):
    def test_model_translations(self):
        """Test translations with each Helsinki model."""
        print('\nTEST HELSINKI MODELS:')
        print('----------------------------------------------------------------------')
        for model_name, source_text in HELSINKI_MODELS.items():
            with self.subTest(model=model_name):
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)

                tokenized_text = tokenizer(source_text, return_tensors='pt')
                generate_translations = model.generate(
                    input_ids=tokenized_text['input_ids'],
                    attention_mask=tokenized_text['attention_mask'],
                )
                translated_text = tokenizer.decode(
                    generate_translations[0], skip_special_tokens=True
                )

                self.assertIsNotNone(translated_text)
                self.assertNotEqual(translated_text, '')

                print(f'\nSource text ({model_name}): {source_text}')
                print(f'\nTranslated text ({model_name}): {translated_text}')

        print('----------------------------------------------------------------------')


if __name__ == '__main__':
    unittest.main()
