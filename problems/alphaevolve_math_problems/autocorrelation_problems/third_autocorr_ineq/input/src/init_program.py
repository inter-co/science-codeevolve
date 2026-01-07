# EVOLVE-BLOCK-START

import numpy as np

def construct_function() -> list[float]:
    """Function to construct step-function with low C3 value."""
    f_values = [np.random.random()] * np.random.randint(100,1000)
    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
