"""Load/save and image processing functions for supporting chess assignment.

DO NOT ALTER THIS FILE.

version: v1.0
"""
import gzip
import json

import itertools

import numpy as np
from PIL import Image

from typing import List

N_SQUARES = 8  # Chessboard is 8x8 squares


def load_square_images(board_image_file: str) -> List[np.ndarray]:
    """Load a board and return as list of board square images.

    Args:
        board_image_file (str): Name of board image file.

    Returns:
        list[np.ndarray]: List of images representing each square of the board.
    """

    im = np.array(Image.open(board_image_file))
    assert im.shape[0] == im.shape[1]  # Square image

    sq_size = int(im.shape[0] / N_SQUARES)

    images = [
        im[
            row * sq_size : (row + 1) * sq_size,
            col * sq_size : (col + 1) * sq_size,
        ]
        for row in range(N_SQUARES)
        for col in range(N_SQUARES)
    ]
    return images


def flatten(list_of_lists):
    """Flatten a list of lists."""
    return list(itertools.chain.from_iterable(list_of_lists))


def load_board_images(image_dir: str, board_data: list) -> List[np.ndarray]:
    """Load a list of board images.

    Args:
        image_dir (str): Name of directory containing board images.
        board_data (list): List of dictionaries contain board metadata.

    Returns:
        list[np.ndarray]: List of square images.
    """

    # Load squares for each board and flatten into single list
    images = flatten(
        [load_square_images(image_dir + "/" + board["image"]) for board in board_data]
    )

    return images


def load_board_labels(board_data: list) -> np.ndarray:
    """Collates the square labels stored in board_data and returns as a single list.

    Args:
        board_data (list): List of dictionaries contain board metadata.

    Returns:
        np.ndarray: List of square labels.
    """

    return np.array(flatten(["".join(board["board"]) for board in board_data]))


def save_jsongz(filename: str, data: dict) -> None:
    """Save a dictionary to a gzipped json file.

    Args:
        filename (str): Name of file to save to.
        data (dict): Dictionary to save.
    """
    with gzip.GzipFile(filename, "wb") as fp:
        json_str = json.dumps(data) + "\n"
        json_bytes = json_str.encode("utf-8")
        fp.write(json_bytes)


def load_jsongz(filename: str) -> dict:
    """Load a gzipped json file.

    Args:
        filename (str): Name of file to load.

    Returns:
        dict: Dictionary loaded from file.
    """
    with gzip.GzipFile(filename, "r") as fp:
        json_bytes = fp.read()
        json_str = json_bytes.decode("utf-8")
        model = json.loads(json_str)
    return model
