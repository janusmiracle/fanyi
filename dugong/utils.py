import psutil
import re

from natsort import natsorted
from pathlib import Path
from typing import List

THRESHOLD = 95.0


def check_memory():
    """Returns True if memory usage is above set threshold."""
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > THRESHOLD:
        return True
    return False


def clean_invalid(name: str) -> str:
    """Removes invalid characters from a string."""
    return re.sub(r"[^\w\s\-\(\)\[\]]", "", name)


def sort_files(path: Path) -> List[Path]:
    """Sort text files in a directory using natural sorting."""
    files = path.iterdir()
    sorted_files = natsorted(files)

    return list(sorted_files)


if __name__ == "__main__":
    check_memory()
