
import numpy as np
import random

class KissRandomNumberGenerator:
    def __init__(self, seed=None):
        """Initialize the RNG with an optional seed."""
        self.seed = seed or random.randint(0, 2**64)
        random.seed(self.seed)

    def uniform_rng(self, min_val=0.0, max_val=1.0):
        """Generate a uniform random number between min_val and max_val."""
        return random.uniform(min_val, max_val)

# Example Usage
if __name__ == "__main__":
    kiss_rng = KissRandomNumberGenerator(seed=12345)
    print("Generating Random Numbers:")
    for _ in range(5):
        print(kiss_rng.uniform_rng(0, 10))  # Example: Random numbers between 0 and 10
