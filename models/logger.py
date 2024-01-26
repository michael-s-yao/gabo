"""
Defines a dummy logging class compatible with MBO logging APIs.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""


class DummyLogger:
    """Defines a dummy logging class."""

    def __init__(self, *args, **kwargs):
        """
        Args:
            None.
        """
        return  # Do nothing.

    def add_scalar(self, *args, **kwargs) -> None:
        """
        Adds a scalar to the log.
        Input:
            None.
        Returns:
            None.
        """
        return  # Do nothing.

    def record(self, *args, **kwargs) -> None:
        """
        Records input information to the log.
        Input:
            None.
        Returns:
            None.
        """
        return  # Do nothing.
