import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        #pass

        # THOUGHT PROCESS
        # numerator first: sum( (true * ln(pred_prob) + (1-true) * ln(1-pred_prob)) )
        # add in small error to avoid log(0)
        eps = 1e-7
        numerator = np.sum(y_true * np.log(y_pred + eps) + (1-y_true) * np.log(1-y_pred+eps))
        # for the * -1/n
        ans = numerator * (-1/len(y_true))
        return round(ans ,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        #pass

        # THOUGHT PROCESS
        # multi class , iterate y_pred properly (class based)
        # numerator first: true * ln(pred_prob)
        eps = 1e-7
        
        numerator = np.sum(y_true * np.log(y_pred + eps))
        ans = numerator * (-1/len(y_true))
        return round(ans,4)
