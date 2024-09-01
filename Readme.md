# Decision Tree Classifier by David Megli

Welcome to the **Decision Tree Classifier** program created by David Megli. This tool leverages datasets from the UCI Machine Learning Repository to train a decision tree classifier using k-fold cross-validation.

## Features

- Fetch datasets directly from the UCI Machine Learning Repository.
- Train a decision tree classifier using stratified k-fold cross-validation.
- Display the list of classes and predictions, highlighting errors in red.
- Report the number of errors and relative error for each fold.

## Usage

1. Run the `main.py` script.
2. Select a dataset from the provided list, or input the dataset ID from the UCI Machine Learning Repository.
3. Enter the desired number of folds for stratified k-fold cross-validation.

The program will output the classes and predictions, with errors highlighted in red. It will also report the number of errors and relative errors for each fold.

## Requirements

- Python 3.x
- Required Python libraries (can be installed via `pip`)

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```
## Running the Program
To run the program, simply execute:
```bash
python main.py
```
Follow the on-screen prompts to select your dataset and specify the number of folds for cross-validation.

## Example Output
The program will display:

- A list of classes and predictions.
- Errors will be highlighted in red.
- Summary of errors and relative errors for each fold.
