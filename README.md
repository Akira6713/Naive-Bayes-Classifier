# Naive Bayes Classifier on Iris Dataset

Author: Eric Ramirez

## Requirements

- scikit-learn
- numpy

## Script Description

This script implements a Gaussian Naive Bayes classifier using the scikit-learn library. The classifier is trained on the Iris dataset to make predictions, and its performance is evaluated with accuracy and classification reports. Additionally, the script calculates and displays the posterior probabilities for the test data.


The script performs the following steps:

1. **Load the Dataset**: Loads the Iris dataset from scikit-learn.
2. **Split the Dataset**: Splits the dataset into training and testing sets.
3. **Train the Model**: Initializes and trains a Gaussian Naive Bayes classifier.
4. **Make Predictions**: Makes predictions on the test set.
5. **Evaluate the Model**: Calculates the accuracy and prints a detailed classification report.
6. **Posterior Probabilities**: Calculates and prints the posterior probabilities for the test data.

## Output

### Classification Report
The script provides a detailed classification report showing precision, recall, f1-score, and support for each class.

### [Image of Classification Report]

![classification_report](https://github.com/Akira6713/Naive-Bayes-Classifier/assets/66973202/c06fe213-3965-4e12-8725-ffa47ce29d96)

### Posterior Probabilities
The script also calculates and displays the posterior probabilities, indicating the model's confidence in its predictions for each test sample.

### [Image of Posterior Probabilities]

![posterior_probabilities](https://github.com/Akira6713/Naive-Bayes-Classifier/assets/66973202/bfaa2c6f-e213-4169-9751-f7a7cc2339ad)


