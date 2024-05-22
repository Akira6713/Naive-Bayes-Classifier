# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate Class Priors
def calculate_class_priors(y):
    class_priors = np.bincount(y) / len(y)
    return class_priors

# Calculate Mean and Variance for each feature per class
def calculate_mean_variance(X, y):
    n_classes = len(np.unique(y))
    n_features = X.shape[1]

    means = np.zeros((n_classes, n_features))
    variances = np.zeros((n_classes, n_features))

    for c in range(n_classes):
        X_c = X[y == c]
        means[c, :] = np.mean(X_c, axis=0)
        variances[c, :] = np.var(X_c, axis=0) + 1e-9  # Adding a small value to avoid division by zero

    return means, variances

# Calculate Gaussian Probability Density Function
def gaussian_pdf(x, mean, var):
    coefficient = 1.0 / np.sqrt(2.0 * np.pi * var)
    exponent = np.exp(-((x - mean) ** 2) / (2.0 * var))
    return coefficient * exponent

# Calculate Posterior Probability for Each Class
def calculate_posterior_probability(X, means, variances, class_priors):
    posterior_probs = []
    for x in X:
        class_probs = []
        for c in range(len(class_priors)):
            likelihood = np.prod(gaussian_pdf(x, means[c], variances[c]))
            posterior = likelihood * class_priors[c]
            class_probs.append(posterior)
        posterior_probs.append(class_probs / np.sum(class_probs))  # Normalize
    return np.array(posterior_probs)

# Calculate class priors
class_priors = calculate_class_priors(y_train)

# Calculate mean and variance for each feature per class
means, variances = calculate_mean_variance(X_train, y_train)

# Calculate posterior probabilities for the test data
posterior_probs = calculate_posterior_probability(X_test, means, variances, class_priors)

# Print the posterior probabilities for the test data
print("\nPosterior Probabilities for the test data:")
print(posterior_probs)

# Compare with scikit-learn's GaussianNB results
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
sklearn_posterior_probs = gnb.predict_proba(X_test)
print("\nPosterior Probabilities using scikit-learn's GaussianNB:")
print(sklearn_posterior_probs)
