"""Train the classifier and save model.

DO NOT ALTER THIS FILE.

usage:
  python3 train.py

version: v1.0
"""

import json

import system
from utils import utils


def train(board_metadata: list, image_dir: str) -> dict:
    """Process training data.

    Take the board metadata and image directory as input. The board metadata
    contains the names of the boards to be used for training and the labels
    for each board. The image directory contains the images for each board.
    Will return the parameters for the trained classifier stored in a dictionary.

    Args:
        board_metadata (list): List of dicts containing board metadata.
        image_dir (str): The root directory for image data.

    Returns:
        dict: Dictionary containing model data that has been learned during training.
    """

    images_train = utils.load_board_images(image_dir, board_metadata)
    labels_train = utils.load_board_labels(board_metadata)

    fvectors_train = system.images_to_feature_vectors(images_train)
    model_data = system.process_training_data(fvectors_train, labels_train)
    return model_data


def main():
    """Train the classifier and save the model."""

    # Load the list of boards that will be used for training.
    with open("data/boards.train.json", "r", encoding="utf-8") as fp:
        board_metadata = json.load(fp)

    # Train a model using the clean data.
    print("Training model with clean data")
    model_data = train(board_metadata, "data/clean")
    utils.save_jsongz("data/model.clean.json.gz", model_data)

    # Train a model using the noisy data.
    print("Training model with noisy data")
    model_data = train(board_metadata, "data/noisy")
    utils.save_jsongz("data/model.noisy.json.gz", model_data)


if __name__ == "__main__":
    main()
