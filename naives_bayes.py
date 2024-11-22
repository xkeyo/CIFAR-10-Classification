# Importing the numpy library
import numpy as np

# Implementation of the custom Gaussian Naive Bayes
class GaussianNaiveBayes:

    # Function to train the Gaussian Naive Bayes model
    # X is the input data (features), y is the target labels
    def fit(self, X, y):
        # Get the number of samples and features in dataset
        num_samples, num_features = X.shape

        # Identify the unique classes in the target labels
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        # Calculate mean, variance, and prior probabilities for each class
        self.mean = np.zeros((num_classes, num_features))
        self.var = np.zeros((num_classes, num_features))
        self.prior = np.zeros(num_classes)

        # Calculate mean, variance, and prior probabilities for each class
        for idx, c in enumerate(self.classes):
            # Filter data for the current class
            X_c = X[y == c]
            # Computing mean, variance, and prior probabilities for each class
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.prior[idx] = X_c.shape[0] / num_samples

    # Function to predict labels for new data
    # X is the input data (features) and returns the predicted labels
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    # Function to predict labels for a single data input 
    # x is the input data (features) 
    def _predict(self, x):
        # Initialize an empty list to store the posterior probabilities
        posteriors = [] 

        # Calculate the posterior probability of each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.prior[idx])
            posterior = np.sum(np.log(self.pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # Return the class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]
    
    # Probability density function (pdf) 
    # class_idx is the index of the class, x is the input data
    def pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-0.5 * np.sum((x - mean)**2 / var))
        denominator = np.sqrt(2 * np.pi * np.prod(var))
        return numerator / denominator