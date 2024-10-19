
# Chessboard Diagram Classifier

## Project Overview

**ChessMate** is an image processing system designed to classify chessboard diagrams. The system analyzes chessboard images, detects the pieces on the board, and returns a sequence of labels representing each square. This project was developed as part of a university assignment, and it involves working with both clean and noisy chessboard images from classic chess books. The classifier supports two modes: isolated square classification and full-board classification, where additional context and chess rules are used to improve accuracy.

## Key Features

- **Isolated Square Classification**: Identifies the chess piece on each square using only the image of that square.
- **Full-board Classification**: Utilizes the full chessboard context, applying chess rules to correct potential misclassifications.
- **Noise Reduction**: Implements Gaussian filtering to handle noise in corrupted chessboard images.
- **Dimensionality Reduction**: Uses PCA (Principal Component Analysis) to reduce image dimensionality while preserving key features.
- **k-Nearest Neighbors Classifier**: Classifies each square based on the kNN algorithm, with custom soft voting logic to handle noisy environments.

## Tech Stack

- **Programming Language**: Python
- **Libraries**: Numpy, Scipy, Standard Python libraries
- **Model**: k-Nearest Neighbors (kNN) with PCA for feature reduction
- **Data**: Chessboard images pre-processed into 50x50 pixel squares

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Zhenyang-Liu/Chessboard-Diagram-Classifier.git
   cd project
   ```

2. **Install dependencies**:
   

3. **Run the training script**:
   ```bash
   python train.py
   ```

4. **Run the evaluation script**:
   ```bash
   python evaluate.py
   ```

## Usage

- **Train the model**: The training script processes the input images, reduces dimensionality, and trains the classifier. Results are saved as `.json.gz` files.
- **Evaluate the model**: The evaluation script tests the classifier on clean and noisy data, reporting the percentage of correctly classified squares and boards.

## Performance

- **High-quality data**:
  - Percentage Squares Correct: 99.4%
  - Percentage Boards Correct: 99.4%
  
- **Noisy data**:
  - Percentage Squares Correct: 97.3%
  - Percentage Boards Correct: 98.1%
