import numpy as np
import math


class MyLogisticRegression():
    """
    Description:
    My personal logistic regression to classify things.
    """

    supported_penalties = ['l2']  # Only 'l2' penalty is considered

    def __init__(self, theta, alpha=0.001, max_iter=1000, penalty='l2', lambda_=1.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty in self.supported_penalties else 0

    def sigmoid_(self, x):
        """
        Compute the sigmoid of a vector.
        Args:
            x: has to be a numpy.ndarray of shape (m, 1).
        Returns:
            The sigmoid value as a numpy.ndarray of shape (m, 1).
            None if x is an empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
        """
        return 1 / (1 + np.exp(-x))

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a vector of dimension m * n.
        Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1.
            None if x or theta are empty numpy.ndarray.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exception.
        """
        m = len(x)  # Number of training examples

        # Add a column of ones to X as the first column
        x_prime = np.concatenate((np.ones((m, 1)), x), axis=1)

        return 1 / (1 + np.exp(-np.dot(x_prime, self.theta)))

    def loss_elem_(self, y_hat, eps=1e-15):
        return np.clip(y_hat, eps, 1 - eps)

    def loss_(self, y, y_hat, eps=1e-15):
        """
        Computes the logistic loss value.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            eps: has to be a float, epsilon (default=1e-15)
        Returns:
            The logistic loss value as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """

        m = y.shape[0]  # Number of samples

        y_hat_update = self.loss_elem_(y_hat, eps)

        # Compute the logistic loss
        loss = -np.sum(y * np.log(y_hat_update) + (1 - y)
                       * np.log(1 - y_hat_update)) / m

        if self.penalty == 'l2':
            regularization_term = self.lambda_ / \
                (2 * m) * np.sum(np.square(self.theta[1:]))
            loss += regularization_term

        return loss

    def gradient_(self, x, y):
        """Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible shapes.
        Args:
            x: has to be an numpy.ndarray, a matrix of shape m * n.
            y: has to be an numpy.ndarray, a vector of shape m * 1.
        Returns:
            The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
            None if x, y, or theta are empty numpy.ndarray.
            None if x, y and theta do not have compatible shapes.
        Raises:
            This function should not raise any Exception.
        """
        # Check if the inputs are non-empty and have compatible shapes
        if x.size == 0 or y.size == 0 or self.theta.size == 0 or \
                x.shape[0] != y.shape[0] or x.shape[1] != self.theta.shape[0]:
            return None

        m = x.shape[0]  # Number of samples

        # Compute the predicted probabilities using the sigmoid function
        y_hat = self.sigmoid_(np.dot(x, self.theta))

        # Compute the gradient vector
        gradient = np.dot(x.T, y_hat - y) / m

        # if self.penalty == 'l2':
        #     gradient[1:] += (self.lambda_ / m) * self.theta[1:]

        return gradient

    def fit_(self, x, y):
        # Add bias term to the feature matrix
        x_with_bias = np.hstack((np.ones((x.shape[0], 1)), x))
        # print(x_with_bias)

        for i in range(self.max_iter):
            gradient_update = self.gradient_(x_with_bias, y)
            if gradient_update is None:
                return None
            # self.theta = self.theta.astype(np.float64)
            # Update theta using the mean gradient
            self.theta -= self.alpha * gradient_update
            # if i % 10000 == 0:
            #     print(i, "th:", self.theta.flatten())
        return self.theta

    def stochastic_fit_(self, x, y, batch=0):
        num_examples = x.shape[0]
        for iteration in range(self.max_iter):
            shuffled_indices = np.random.permutation(num_examples)
            x_shuffled = x[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            for i in range(num_examples):
                x_i = x_shuffled[i]
                y_i = y_shuffled[i]

                gradient_update = self.gradient_(
                    x_i[np.newaxis, :batch], y_i[np.newaxis, :batch])
                if gradient_update is None:
                    return None

                self.theta = self.theta.astype(np.float64)
                self.theta -= self.alpha * \
                    gradient_update.mean(axis=1, keepdims=True)

            if iteration % 10000 == 0:
                print(iteration, "th:", self.theta.flatten())
        return self.theta

    # stochastic_fit testing without having inner loop
    def stochastic_fit_2_(self, x, y):
        num_examples = x.shape[0]
        for iteration in range(self.max_iter):
            shuffled_indices = np.random.permutation(num_examples)
            x_shuffled = x[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            gradient_update = self.gradient_(
                # Pick the first random data point
                x_shuffled[:1], y_shuffled[:1]
            )
            if gradient_update is None:
                return None

            self.theta = self.theta.astype(np.float64)
            # Update theta using the mean gradient
            self.theta -= self.alpha * \
                gradient_update.mean(axis=1, keepdims=True)

            if iteration % 10000 == 0:
                print(iteration, "th:", self.theta.flatten())
        return self.theta
