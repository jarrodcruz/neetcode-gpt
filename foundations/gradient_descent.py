class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        minimizer = init # x value

        for _ in range(iterations):
            deriv = 2 * minimizer # derivative of x^2 = 2x
            minimizer = minimizer - learning_rate * deriv # guess = guess - alpha * d

        return round(minimizer,5)