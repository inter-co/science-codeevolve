# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    f_values = [np.random.random()] * np.random.randint(100,1000)
    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
