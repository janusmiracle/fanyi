import os
import unittest
from pathlib import Path

from fanyi.errors import ValidationError
from fanyi.imports import import_raws, import_translations

TEST_RAWS_CH = Path(os.getcwd() + '/tests/custom_dataset/chinese/raws/')
TEST_TRANSLATIONS_CH = Path(os.getcwd() + '/tests/custom_dataset/chinese/translations/')


class TestImportRaws(unittest.TestCase):
    def test_valid_input(self):
        output_directory = import_raws(TEST_RAWS_CH, 'Chinese-Raws')
        self.assertIsInstance(output_directory, Path)
        if output_directory is not None:
            self.assertTrue(output_directory.exists())
            self.assertTrue(any(output_directory.glob('*.txt')))

    def test_invalid_source_name_not_cleaned(self):
        with self.assertRaises(ValidationError):
            import_raws(TEST_RAWS_CH, 'INVALID/NAME')

    def test_invalid_source_name_cleaned(self):
        try:
            import_raws(TEST_RAWS_CH, '\\^^^,,//~~~Chinese-Raws\\2', auto_clean=True)
        except ValidationError:
            self.fail('ValidationError should not be raised when auto_clean is True')

    def test_invalid_source_directory(self):
        with self.assertRaises(ValidationError):
            import_raws(Path('invalid/path'), 'Chinese-Raws')


class TestImportTranslations(unittest.TestCase):
    def test_valid_input(self):
        output_directory = import_translations(
            TEST_TRANSLATIONS_CH, 'Chinese-Translations'
        )
        self.assertIsInstance(output_directory, Path)
        if output_directory is not None:
            self.assertTrue(output_directory.exists())
            self.assertTrue(any(output_directory.glob('*.txt')))

    def test_invalid_source_name_not_cleaned(self):
        with self.assertRaises(ValidationError):
            import_translations(TEST_TRANSLATIONS_CH, 'INVALID/NAME')

    def test_invalid_source_name_cleaned(self):
        try:
            import_translations(
                TEST_TRANSLATIONS_CH,
                '\\^^^,,//~~~Chinese-Translations\\[]',
                auto_clean=True,
            )
        except ValidationError:
            self.fail('ValidationError should not be raised when auto_clean is True')

    def test_invalid_source_directory(self):
        with self.assertRaises(ValidationError):
            import_translations(Path('invalid/path'), 'Chinese-Translations')


if __name__ == '__main__':
    unittest.main()
