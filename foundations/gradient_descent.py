class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        #pass
        x = init
        for i in range(iterations):
            x = self.update_rule(x, learning_rate)
        return round(x,5)
    
    def update_rule(self, x: int, learning_rate: float):
        return x - learning_rate * 2 * x
