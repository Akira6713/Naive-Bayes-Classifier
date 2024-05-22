# Naive Bayes Classifier on Iris Dataset

Author: Eric Ramirez

## Requirements

- Python 3.x
- scikit-learn
- numpy

## Script Description

This script implements a Gaussian Naive Bayes classifier using the scikit-learn library. The classifier is trained on the Iris dataset to make predictions, and its performance is evaluated with accuracy and classification reports. Additionally, the script calculates and displays the posterior probabilities for the test data.

The script performs the following steps:

1. **Load the Dataset**: Loads the Iris dataset from scikit-learn.
2. **Split the Dataset**: Splits the dataset into training and testing sets.
3. **Calculate Class Priors**: Computes the prior probabilities for each class.
4. **Calculate Mean and Variance**: Calculates the mean and variance for each feature per class.
5. **Gaussian PDF**: Uses the Gaussian probability density function to compute the likelihood of each feature value given the class.
6. **Calculate Posterior Probabilities**: Computes the posterior probabilities for each class using the Gaussian PDF and class priors.
7. **Make Predictions**: Makes predictions on the test set using the calculated posterior probabilities.
8. **Evaluate the Model**: Calculates the accuracy and prints a detailed classification report.
9. **Compare with scikit-learn's GaussianNB**: Compares the custom implementation's posterior probabilities with those obtained using scikit-learn's GaussianNB to validate the custom implementation.

## Output

### Posterior Probabilities for the Test Data
The script calculates and displays the posterior probabilities, indicating the model's confidence in its predictions for each test sample.

![posterior_probabilities_test_data](https://github.com/Akira6713/Naive-Bayes-Classifier/assets/66973202/a5a54c3d-ebf0-4a5b-9934-3ea57cba18d9)

### Posterior Probabilities using scikit-learn's GaussianNB
The script also compares the posterior probabilities with those obtained using scikit-learn's GaussianNB to validate the custom implementation.

![posterior_probabilities_gaussian_nb](https://github.com/Akira6713/Naive-Bayes-Classifier/assets/66973202/edd31e9f-79ba-4a1b-9221-d4c32ae811d2)

