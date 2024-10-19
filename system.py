"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from collections import Counter
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

N_DIMENSIONS = 10


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    labels = []
    for test_vector in test:
        # Calculate the Euclidean distance between the test vector and all training vectors
        distances = np.linalg.norm(train - test_vector, axis=1)

        # Find the index of the k smallest distances
        k_neighbors_indices = np.argpartition(distances, 5)[:5]

        # Get k nearest neighbour labels
        k_neighbor_labels = train_labels[k_neighbors_indices]
        k_neighbor_distances = distances[k_neighbors_indices]

        # Compute Gaussian weights based on distances
        weights = norm.pdf(k_neighbor_distances, scale=300)

        # Aggregate weighted voting
        total_weight = 0
        label_votes = Counter()
        for label, weight in zip(k_neighbor_labels, weights):
            label_votes[label] += weight
            total_weight += weight
        # Identify the labels with the highest weighted votes
        sorted_votes = label_votes.most_common()
        most_common_label, highest_vote = sorted_votes[0]

        # If the highest vote is a space tag, check the gap to the second-highest vote
        if most_common_label == '.' and len(sorted_votes) > 1:
            second_most_common_label, second_highest_vote = sorted_votes[1]
            if highest_vote - second_highest_vote < 0.12 * total_weight:
                most_common_label = second_most_common_label

        labels.append(most_common_label)
    return labels


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    # Get PCA parameters from the model
    mean = model["mean"]
    principal_components = model["principal_components"]

    # Centred data
    data_centered = data - mean

    # Apply the PCA model
    reduced_data = np.dot(data_centered, principal_components)

    return reduced_data


def train_pca_model(data: np.ndarray, labels: np.ndarray) -> dict:
    """Train a PCA model on the given data."""
    # Calculate the mean of the data and centre the data
    mean = np.mean(data, axis=0)
    data_centered = data - mean

    # Calculate the covariance matrix
    covariance_matrix = np.cov(data_centered, rowvar=False)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Select the eigenvector corresponding to the largest N_DIMENSIONS eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    principal_components = eigenvectors[:, idx[:N_DIMENSIONS*2]]

    # Calculate 1-dimensional divergence for each principal component
    divergences = []
    for i in range(N_DIMENSIONS*2):
        divergence = 0
        unique_labels = np.unique(labels)
        for class1 in unique_labels:
            data1 = data_centered[labels == class1] @ principal_components[:, i]
            for class2 in unique_labels:
                if class1 != class2:
                    data2 = data_centered[labels == class2] @ principal_components[:, i]
                    divergence += compute_1d_divergence(data1, data2)
        divergences.append(divergence)

    # Select the N_DIMENSIONS principal components with the highest divergence from the inverted list of eigenvalues.
    divergences_idx = np.argsort(divergences)[::-1]
    selected_principal_components = principal_components[:, divergences_idx[:N_DIMENSIONS]]

    # Create PCA models
    pca_model = {
        "mean": mean,
        "principal_components": selected_principal_components
    }

    return pca_model


def compute_1d_divergence(class1: np.ndarray, class2: np.ndarray) -> float:
    """compute a vector of 1-D divergences

    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2

    returns: d12 - a vector of 1-D divergence scores
    """

    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (
            1.0 / v1 + 1.0 / v2
    )

    return d12


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """
    model = {"labels_train": labels_train.tolist()}

    # Train the PCA model and store it in the model dictionary
    pca_model = train_pca_model(fvectors_train, labels_train)
    model["mean"] = pca_model["mean"].tolist()
    model["principal_components"] = pca_model["principal_components"].tolist()

    # Use PCA models to downscale training data
    fvectors_train_reduced = reduce_dimensions(fvectors_train, pca_model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()

    return model


def images_to_feature_vectors(images, sigma=1.185, focus_area=(43, 40)):
    """Applies Gaussian filter to a list of square images, and extracts a centered
    sub-area as feature vectors.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.
        sigma (float): The standard deviation for the Gaussian kernel.
        focus_area (tuple): The size of the central area to extract as feature vector.

    Returns:
        np.ndarray: A 2-D array in which the rows represent feature vectors from
        the central focus area of each image.
    """
    # The number of features is the number of pixels in the focus area
    n_features = focus_area[0] * focus_area[1]
    feature_vectors = np.empty((len(images), n_features))

    # Calculate the border sizes based on the focus area size
    border_size_x = (images[0].shape[0] - focus_area[0]) // 2
    border_size_y = (images[0].shape[1] - focus_area[1]) // 2

    for i, image in enumerate(images):
        # Apply Gaussian filter to the image
        filtered_image = gaussian_filter(image, sigma=sigma)
        # Extract the focus area from the filtered image and flatten it
        focus_area_image = filtered_image[
                           border_size_x:border_size_x + focus_area[0],
                           border_size_y:border_size_y + focus_area[1]
                           ].flatten()

        # Assign the flattened focus area to the feature vector array
        feature_vectors[i, :] = focus_area_image

    return feature_vectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on an array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on an array of image feature vectors presented in 'board order'.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    # Extract training data and labels from the model
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Determine the number of boards
    num_boards = fvectors_test.shape[0] // 64
    assert fvectors_test.shape[0] % 64 == 0

    all_labels = []

    # Classify each board
    for i in range(num_boards):
        board_fvectors = fvectors_test[i * 64:(i + 1) * 64, :]

        # Use classify_with_confidence to get labels and confidence
        board_labels, board_confidences, neighbors_labels_list, neighbors_distances_list = classify_with_confidence(
            fvectors_train, labels_train, board_fvectors)

        # Apply rules to correct classification results
        corrected_board_labels = apply_chess_rules(board_labels, board_confidences, neighbors_labels_list,
                                                   neighbors_distances_list)

        all_labels.extend(corrected_board_labels)

    return all_labels


def classify_with_confidence(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray, k: int = 5) -> Tuple[
     List[str], List[float], List[List[str]], List[List[float]]]:
    """
    Classify test instances using k-nearest neighbors and return both labels and confidences.
    
    The confidences are calculated based on the weighted vote, where weight is the inverse of distance.
    
    Args:
        train (np.ndarray): The training feature vectors.
        train_labels (np.ndarray): The training labels.
        test (np.ndarray): The test feature vectors to classify.
        k (int): The number of neighbors to use.
    
    Returns:
        Tuple[List[str], List[float], List[List[str]], List[List[float]]]: The list of predicted labels, 
        their corresponding confidences, neighbors labels and distances.
    """
    predicted_labels = []
    confidences = []
    neighbors_labels_list = []
    neighbors_distances_list = []

    for test_vector in test:
        distances = np.linalg.norm(train - test_vector, axis=1)
        k_neighbor_indices = np.argsort(distances)[:k]
        k_neighbor_labels = train_labels[k_neighbor_indices]
        k_neighbor_distances = distances[k_neighbor_indices]

        # Compute Gaussian weights based on distances
        weights = norm.pdf(k_neighbor_distances, scale=130)
        total_weight = 0
        label_votes = Counter()
        for label, weight in zip(k_neighbor_labels, weights):
            label_votes[label] += weight
            total_weight += weight

        sorted_votes = label_votes.most_common()
        most_common_label, highest_vote = sorted_votes[0]

        if most_common_label == '.' and len(sorted_votes) > 1:
            second_most_common_label, second_highest_vote = sorted_votes[1]
            if highest_vote - second_highest_vote < 0.12 * total_weight:
                most_common_label = second_most_common_label

        # Calculate the confidence as the weighted score normalized by the total weight
        confidence = highest_vote / total_weight

        # Store the results
        predicted_labels.append(most_common_label)
        confidences.append(confidence)
        neighbors_labels_list.append(k_neighbor_labels.tolist())
        neighbors_distances_list.append(k_neighbor_distances.tolist())

    return predicted_labels, confidences, neighbors_labels_list, neighbors_distances_list


def apply_chess_rules(board_labels, board_confidences, neighbors_labels_list, neighbors_distances_list):
    """
        Applies a series of chess-specific rules to correct potential misclassifications on the chessboard.

        This function sequentially applies various chess rules to ensure the accuracy of the
        classification. It starts by checking and correcting the presence of kings. Then, it flags
        suspicious pieces based on the position of pawns and the total number of each type of piece.
        After flagging, it processes these pieces for potential reassessment. Finally, it checks
        that bishops are on different colored squares and, if necessary, flags and reassesses them.
        The function iteratively updates both the board labels and their confidences throughout the process.

        Args:
            board_labels (list): Labels of each square on the chessboard.
            board_confidences (list): Confidence scores corresponding to each label.
            neighbors_labels_list (list): Lists of neighbor labels for each square.
            neighbors_distances_list (list): Lists of distances to neighbors for each square.

        Returns:
            list: Updated board labels after applying the chess rules, reflecting more accurate classifications.
    """
    # Check the presence of the king
    corrected_board, corrected_confidences = check_and_correct_kings(board_labels, board_confidences)

    # Checking the position of pawns and the number of various pieces and flagging suspicious pieces
    flags, impossible_pieces = pawn_and_totalNum_check(corrected_board, corrected_confidences)
    # Processing labelled pieces and updating the board and confidence
    corrected_board, corrected_confidences = process_flagged_pieces(corrected_board, corrected_confidences, flags,
                                                                    neighbors_labels_list, neighbors_distances_list,
                                                                    impossible_pieces)

    # Check that two bishops on the same side are on different colours
    flags = check_bishops_on_same_color(corrected_board, corrected_confidences)
    # Processing labelled pieces and updating the board and confidence
    corrected_board, corrected_confidences = process_flagged_pieces(corrected_board, corrected_confidences, flags,
                                                                    neighbors_labels_list, neighbors_distances_list)

    return corrected_board


def process_flagged_pieces(board_labels, board_confidences, flags, neighbors_labels_list, neighbors_distances_list,
                           impossible_labels=None):
    """
       Processes flagged chess pieces on the board for reassessment based on neighbors' labels
       and distances, considering any 'impossible' labels.

       This function iterates over each square on the chessboard. If a piece is flagged for
       reassessment, it reevaluates the piece's label by considering the labels and distances of
       its neighboring squares, excluding any labels deemed 'impossible'. The reassessment
       uses a weighted voting system, where closer neighbors have more influence. If the current
       label's confidence is below a certain threshold, the function selects a new label with the
       highest weighted vote. Additionally, special handling is included for cases where the
       space label ('.') might not be the best choice despite having the highest vote.

       Args:
           board_labels (list): Labels of each square on the chessboard.
           board_confidences (list): Confidence scores corresponding to each label.
           flags (numpy.ndarray): A 2D array of flags indicating squares to be reassessed.
           neighbors_labels_list (list): Lists of neighbor labels for each square.
           neighbors_distances_list (list): Lists of distances to neighbors for each square.
           impossible_labels (list, optional): Labels that should not be considered. Defaults to None.

       Returns:
           Tuple[list, list]: Updated board labels and confidences, each flattened into a 1D list.
    """
    board_size = 8
    # Reshape one-dimensional arrays into two-dimensional board
    board = np.array(board_labels).reshape(board_size, board_size)
    confidences_reshaped = np.array(board_confidences).reshape(board_size, board_size)

    if impossible_labels is None:
        impossible_labels = []

    # Handling of each flagged piece
    for row in range(board_size):
        for col in range(board_size):
            if flags[row, col]:
                index = row * board_size + col
                neighbors_labels = neighbors_labels_list[index]
                neighbors_distances = neighbors_distances_list[index]

                # Logic of reassessment
                weighted_label_counts = Counter()
                total_weight = 0 + 1e-5
                for label, distance in zip(neighbors_labels, neighbors_distances):
                    if label not in impossible_labels:
                        weight = 1 / (distance + 1e-5)
                        weighted_label_counts[label] += weight
                        total_weight += weight

                sorted_labels = weighted_label_counts.most_common()
                current_label = board[row, col]
                current_label_weight = weighted_label_counts[current_label]

                best_label = current_label
                best_weight = current_label_weight
                if current_label_weight / total_weight < 0.8:
                    # Select a different label from the current one
                    for label, weight in sorted_labels:
                        if label != current_label:
                            best_label = label
                            best_weight = weight
                            break
                        else:
                            continue

                # Special case: if the best label is a space and the difference with the second-best label < 15%
                if best_label == '.' and len(sorted_labels) > 1:
                    second_best_label, second_best_weight = sorted_labels[1]
                    if best_weight - second_best_weight < 0.15 * total_weight:
                        best_label = second_best_label
                        best_weight = second_best_weight

                confidence = best_weight / total_weight
                board[row, col] = best_label
                confidences_reshaped[row, col] = confidence

    return board.flatten().tolist(), confidences_reshaped.flatten().tolist()


def pawn_and_totalNum_check(board_labels, board_confidences):
    """
        Checks the chessboard for any excess pieces beyond standard limits and pawns in
        inappropriate rows.

        This function ensures that the count of each type of chess piece (king, queen, rook,
        knight, bishop, pawn) doesn't exceed the standard limits (e.g., each side should have
        only one king). It also checks that pawns are not placed in the first or last rows, as
        this is against chess rules. Pieces that exceed the count or pawns in wrong positions
        are flagged for reassessment.

        Args:
            board_labels (list): A list of labels for each square on the chessboard.
            board_confidences (list): A list of confidence scores corresponding to the labels.

        Returns:
            Tuple[numpy.ndarray, list]: A tuple where the first element is a 2D array of flags
                                        (True indicating pieces to be reassessed), and the second
                                        element is a list of pieces that have reached their limit
                                        and are considered 'impossible' to appear again.
    """
    board_size = 8
    board = np.array(board_labels).reshape(board_size, board_size)
    confidences_reshaped = np.array(board_confidences).reshape(board_size, board_size)

    # Check that the number of various pieces does not exceed the limit
    flags = np.zeros_like(board, dtype=bool)
    counts = {'k': 1, 'q': 1, 'r': 2, 'n': 2, 'b': 2, 'p': 8}
    current_counts = {piece.upper(): 0 for piece in counts}
    current_counts.update({piece.lower(): 0 for piece in counts})

    for piece in counts:
        for color in [piece.upper(), piece.lower()]:
            piece_positions = np.where(board == color)
            piece_count = len(piece_positions[0])
            current_counts[color] += piece_count

            if piece_count > counts[piece]:
                piece_confidences = confidences_reshaped[piece_positions]
                low_confidence_indices = np.argsort(piece_confidences)[:piece_count - counts[piece]]
                flags[piece_positions[0][low_confidence_indices], piece_positions[1][low_confidence_indices]] = True

    # Check if pawns appear in the first or last rows
    flags[0, board[0, :] == 'p'] = True
    flags[0, board[0, :] == 'P'] = True
    flags[7, board[7, :] == 'p'] = True
    flags[7, board[7, :] == 'P'] = True

    # The types of pieces that have reached their limit are added to the list of impossibilities.
    impossible_pieces = [color for color, piece_count in current_counts.items() if piece_count >= counts[color.lower()]]

    return flags, impossible_pieces


def check_and_correct_kings(board_labels, board_confidences):
    """
      Checks for the presence of kings on the chessboard for both sides and corrects any
      misclassifications if a king is missing.

      In a standard chess game, each side (White and Black) must have one king. This function
      checks if either side's king is missing from the board. If a king is missing, the function
      attempts to identify a queen (from the first or last row, or any other position if none are
      in those rows) with the lowest confidence score and reclassifies it as a king. This
      correction is based on the observation that queens are often misclassified as kings.

      Args:
          board_labels (list): A list of labels for each square on the chessboard.
          board_confidences (list): A list of confidence scores corresponding to the labels.

      Returns:
          Tuple[List[str], List[float]]: The updated list of labels and confidence scores for
                                         each square on the chessboard, flattened into 1D lists.
    """
    board_size = 8
    board = np.array(board_labels).reshape(board_size, board_size)
    confidences_reshaped = np.array(board_confidences).reshape(board_size, board_size)

    # Check White's and Black's kings
    white_king_positions = np.where(board == 'K')
    black_king_positions = np.where(board == 'k')

    # If White has no king
    if len(white_king_positions[0]) == 0:
        # Prioritise the first and last rows of queens
        white_queen_positions_first_last_row = np.where((board == 'Q') & (
                (np.arange(board_size)[:, None] == 0) | (np.arange(board_size)[:, None] == board_size - 1)))
        if len(white_queen_positions_first_last_row[0]) > 0:
            white_queen_positions = white_queen_positions_first_last_row
        else:
            # If there are no queens in the first and last rows, consider queens in other positions on the board
            white_queen_positions = np.where(board == 'Q')

        if len(white_queen_positions[0]) > 0:
            lowest_confidence_index = np.argmin(confidences_reshaped[white_queen_positions])
            queen_row, queen_col = white_queen_positions[0][lowest_confidence_index], white_queen_positions[1][
                lowest_confidence_index]
            board[queen_row, queen_col] = 'K'
            confidences_reshaped[queen_row, queen_col] = 1.0

    # If Black has no king
    if len(black_king_positions[0]) == 0:
        black_queen_positions_first_last_row = np.where((board == 'q') & (
                (np.arange(board_size)[:, None] == 0) | (np.arange(board_size)[:, None] == board_size - 1)))
        if len(black_queen_positions_first_last_row[0]) > 0:
            black_queen_positions = black_queen_positions_first_last_row
        else:
            black_queen_positions = np.where(board == 'q')

        if len(black_queen_positions[0]) > 0:
            lowest_confidence_index = np.argmin(confidences_reshaped[black_queen_positions])
            queen_row, queen_col = black_queen_positions[0][lowest_confidence_index], black_queen_positions[1][
                lowest_confidence_index]
            board[queen_row, queen_col] = 'k'
            confidences_reshaped[queen_row, queen_col] = 1.0

    return board.flatten().tolist(), confidences_reshaped.flatten().tolist()


def check_bishops_on_same_color(board_labels, board_confidences):
    """
       Checks each side's bishops on the chessboard to ensure they are on different colored squares.

       In chess, each player has two bishops, one on a white square and the other on a black square.
       This function verifies that for both white ('B') and black ('b') bishops, they are not
       positioned on the same color square. If they are, the one with the lower confidence score
       is flagged for reassessment.

       Args:
           board_labels (list): A list of labels for each square on the chessboard.
           board_confidences (list): A list of confidence scores corresponding to the labels.

       Returns:
           numpy.ndarray: A 2D array of boolean flags where True indicates a bishop that
                          may be misclassified and should be reassessed.
    """
    board_size = 8
    board = np.array(board_labels).reshape(board_size, board_size)
    confidences_reshaped = np.array(board_confidences).reshape(board_size, board_size)

    flags = np.zeros((board_size, board_size), dtype=bool)

    # For each side ('B' and 'b'), check the bishop's position
    for color in ['B', 'b']:
        bishop_positions = np.where(board == color)
        if len(bishop_positions[0]) == 2:
            # Check that the two bishops are on the same colour grid
            pos1 = (bishop_positions[0][0], bishop_positions[1][0])
            pos2 = (bishop_positions[0][1], bishop_positions[1][1])
            if (pos1[0] + pos1[1]) % 2 == (pos2[0] + pos2[1]) % 2:
                # If yes, identify the elephant with the lower confidence
                if confidences_reshaped[pos1] < confidences_reshaped[pos2]:
                    flags[pos1] = True
                else:
                    flags[pos2] = True

    return flags
