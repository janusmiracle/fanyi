import os
import unittest
import shutil
from pathlib import Path

from fanyi.file_manager import import_raws, import_translations
from fanyi.errors import ValidationError

TEST_RAWS = Path(os.getcwd() + "/tests/test_files/raws/")
TEST_TRANSLATIONS = Path(os.getcwd() + "/tests/test_files/translations/")


class TestImportRaws(unittest.TestCase):
    def tearDown(self) -> None:
        raws_directory = Path("fanyi") / "raws"
        for directory in [raws_directory]:
            if directory.exists() and directory.is_dir():
                for item in directory.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)

    def test_valid_input(self):
        output_directory = import_raws(TEST_RAWS, "TEST-RAWS")
        self.assertIsInstance(output_directory, Path)
        if output_directory is not None:
            self.assertTrue(output_directory.exists())
            self.assertTrue(any(output_directory.glob("*.txt")))

    def test_invalid_source_name_not_cleaned(self):
        with self.assertRaises(ValidationError):
            import_raws(TEST_RAWS, "INVALID/NAME")

    def test_invalid_source_name_cleaned(self):
        try:
            import_translations(TEST_RAWS, "\\^^^,,//~~~TEST-RAWS\\[]", auto_clean=True)
        except ValidationError:
            self.fail("ValidationError should not be raised when auto_clean is True")

    def test_invalid_source_directory(self):
        with self.assertRaises(ValidationError):
            import_raws(Path("invalid/path"), "TEST-RAWS")


class TestImportTranslations(unittest.TestCase):
    def tearDown(self) -> None:
        translations_directory = Path("fanyi") / "translations"

        for directory in [translations_directory]:
            if directory.exists() and directory.is_dir():
                for item in directory.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)

    def test_valid_input(self):
        output_directory = import_translations(TEST_TRANSLATIONS, "TEST-TRANSLATIONS")
        self.assertIsInstance(output_directory, Path)
        if output_directory is not None:
            self.assertTrue(output_directory.exists())
            self.assertTrue(any(output_directory.glob("*.txt")))

    def test_invalid_source_name_not_cleaned(self):
        with self.assertRaises(ValidationError):
            import_translations(TEST_TRANSLATIONS, "INVALID/NAME")

    def test_invalid_source_name_cleaned(self):
        try:
            import_translations(
                TEST_TRANSLATIONS, "\\^^^,,//~~~TEST-TRANSLATIONS\\[]", auto_clean=True
            )
        except ValidationError:
            self.fail("ValidationError should not be raised when auto_clean is True")

    def test_invalid_source_directory(self):
        with self.assertRaises(ValidationError):
            import_translations(Path("invalid/path"), "TEST-TRANSLATIONS")


if __name__ == "__main__":
    unittest.main()
