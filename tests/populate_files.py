import os

TEST_RAW_PATH = os.getcwd() + '/tests/custom_dataset/chinese/raws/'
TEST_TRANSLATIONS_PATH = os.getcwd() + '/tests/custom_dataset/chinese/translations/'


def populate_test_files() -> None:
    """
    -- DO NOT USE --
    """

    def create_files(folder, name):
        for i in range(1, 11):
            filename = f'MHOH-{name}-{i:03}.txt'
            file_path = os.path.join(folder, filename)
            with open(file_path, 'w') as file:
                file.write(f' ')

    create_files(TEST_RAW_PATH, 'RAWS')
    create_files(TEST_TRANSLATIONS_PATH, 'TRANSLATIONS')


if __name__ == '__main__':
    populate_test_files()
