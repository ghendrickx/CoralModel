"""
Package-specific exceptions.

Author: Gijs G. Hendrickx
"""


class DataError(Exception):
    """Error indicates missing data that is required."""
    pass


class InitialisationError(Exception):
    """Error indicates a problem with initialisation."""
    pass
