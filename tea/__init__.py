"""The Embedding Alphabet (TEA) package."""

from pathlib import Path


def get_matrix_path() -> Path:
    """
    Get the path to the matcha.out substitution matrix file.

    This file can be used with MMseqs2 as a substitution matrix for tea sequences.

    Returns:
        Path: Absolute path to the matcha.out file

    Example:
        >>> from tea import get_matrix_path
        >>> matcha_path = get_matrix_path()
    """
    return Path(__file__).parent / "matcha.out"


__all__ = ["get_matrix_path"]
