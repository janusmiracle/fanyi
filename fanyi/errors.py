class ValidationError(Exception):
    """Exception raised for validation errors."""

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors if errors is not None else []

    def __str__(self):
        error_string = "\n".join([f"- {error}" for error in self.errors])
        return f"{super().__str__()}\n" f"Errors:\n" f"{error_string}"


class InvalidCharacterError(Exception):
    """Exception raised for invalid characters in a string."""

    def __init__(self, message):
        super().__init__(message)


class EmptyDirectoryError(Exception):
    """Exception raised for empty directories."""

    def __init__(self, message):
        super().__init__(message)
