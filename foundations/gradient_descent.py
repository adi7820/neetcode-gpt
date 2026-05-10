class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        new_x = init
        for _ in range(iterations):
            d = 2*new_x
            new_x = new_x - learning_rate * d

        return round(new_x,5)
    