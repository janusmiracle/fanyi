import unittest
from fanyi.utils import clean_invalid


class TestUtils(unittest.TestCase):
    def test_clean_invalid(self):
        # Test with valid input
        valid_input = "valid_name"
        self.assertEqual(clean_invalid(valid_input), valid_input)

        # Test with invalid characters
        invalid_input = "inval!d_n@me"
        self.assertNotEqual(clean_invalid(invalid_input), "invalid_name")

        # Test with an input that contains invalid characters, but after cleaning,
        # the resulting string is equal to a valid input
        invalid_input = "i@@nval!id_na^^^&me////"
        self.assertEqual(clean_invalid(invalid_input), "invalid_name")


# Add other util tests when they are created

if __name__ == "__main__":
    unittest.main()
