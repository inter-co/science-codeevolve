# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass
# Added for multi-scale interpolation (jnp.interp is used instead of scipy.interpolate.interp1d for JAX compatibility)

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    stage: str = "Default" # Added for multi-stage optimization (inspired by IP3)
    num_intervals: int = 1024 # Increased resolution (power of 2 for FFT efficiency, higher than IP1, adapted from IP2/IP3)
    learning_rate: float = 0.003 # Learning rate from IP2
    num_steps: int = 150000 # Increased steps for higher resolution and more complex landscapes (from IP2)
    warmup_steps: int = 15000 # Adjusted warmup steps (scaled from IP2)
    tv_reg_coeff: float = 1e-6 # Coefficient for Total Variation regularization (tuned down from IP3)
    l2_reg_coeff: float = 1e-7 # Coefficient for L2 regularization (tuned down from IP3)

class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]: # Changed return type for regularization
        """
        Computes the objective function using the unitless norm calculation.
        Assumes f is a piecewise constant function on [0, 1].
        """
        f_non_negative = jax.nn.relu(f_values)
        
        N = self.hypers.num_intervals
        dx = 1.0 / N # Discretization step size for f on [0, 1]
        
        # 1. Calculate ∫f dx
        integral_f = jnp.sum(f_non_negative) * dx
        
        # 2. Compute convolution g = f ★ f
        # Pad f for FFT-based convolution. If f has N points on [0,1],
        # padded_f length 2N ensures convolution samples g(x) on [0, 2] at 2N points.
        padded_f = jnp.pad(f_non_negative, (0, N)) # Padded_f has length 2N
        fft_f = jnp.fft.fft(padded_f)
        raw_convolution_ifft = jnp.fft.ifft(fft_f * fft_f).real # Convolution has length 2N
        
        # Correctly scale the convolution samples (critical fix, inspired by IP2)
        # jnp.fft.ifft scales by 1/len(input). To get samples of (f*f)(x) * dx,
        # we need to multiply by len(input) * dx. Here, len(input) = 2N, dx = 1/N.
        # So, scaling factor is 2N * (1/N) = 2.0.
        convolution_samples = raw_convolution_ifft * 2.0
        
        # Ensure convolution remains non-negative due to potential numerical noise from IFFT
        convolution_samples = jax.nn.relu(convolution_samples)

        # The convolution 'g' has 2N samples and spans the domain [0, 2].
        # The discretization step for the convolution domain is dx_conv = 2.0 / (2N) = 1.0 / N = dx.
        dx_conv = dx
        
        # 3. Calculate ||g||_2^2 = ∫g(x)^2 dx
        # Using Riemann sum for simplicity and robustness (inspired by IP2).
        l2_norm_squared = jnp.sum(convolution_samples**2) * dx_conv

        # 4. Calculate ||g||_inf = sup |g(x)|
        norm_inf = jnp.max(convolution_samples)
        
        # 5. Calculate C2 ratio using the formula: C2 = ||g||_2^2 / ( (∫f)^2 * ||g||_inf )
        int_f_squared = integral_f**2
        
        # Handle potential division by zero. Add a small epsilon for numerical stability.
        # It's crucial that integral_f_squared is not zero and norm_inf is not zero.
        denominator = int_f_squared * norm_inf
        denominator = jnp.where(denominator < 1e-12, 1e-12, denominator) # Use a threshold, not just == 0
        
        c2_ratio = l2_norm_squared / denominator
        
        # Compute Regularization terms (inspired by IP3, but with much smaller coeffs)
        # TV regularization: sum of absolute differences between adjacent f_values
        tv_regularization = jnp.sum(jnp.abs(jnp.diff(f_non_negative)))
        # L2 regularization: sum of squared f_values
        l2_regularization = jnp.sum(f_non_negative**2) * dx # Integrate f^2 over [0,1]

        # Combine into final objective value. We want to MAXIMIZE C2, so MINIMIZE -C2.
        # Add regularization terms to the objective.
        objective_value = -c2_ratio + \
                          self.hypers.tv_reg_coeff * tv_regularization + \
                          self.hypers.l2_reg_coeff * l2_regularization
        
        return objective_value, c2_ratio # Return both for logging

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        # Modified to handle has_aux=True for objective_fn (inspired by IP3)
        (loss, c2_val), grads = jax.value_and_grad(self._objective_fn, has_aux=True)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss, c2_val # Return c2_val for logging

    def run_optimization(self, initial_f_values: jnp.ndarray | None = None): # Added initial_f_values for warm-starting
        """Sets up and runs the full optimization process."""
        print(f"\n--- Starting Optimization Stage: {self.hypers.stage} ---") # Added stage info
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-4
        )
        self.optimizer = optax.adam(learning_rate=schedule)
        
        if initial_f_values is None:
            print("Initializing with new random values for f_values.")
            key = jax.random.PRNGKey(42)
            key, subkey = jax.random.split(key) # Use subkey for f_values
            # Initialize f_values with small random uniform values to encourage diverse exploration
            # and prevent symmetry issues that a constant initialization might introduce.
            # Values are kept positive (minval=0.01) to align with f(x) >= 0 constraint and ensure non-trivial starting points. (inspired by IP2)
            f_values = jax.random.uniform(subkey, (self.hypers.num_intervals,), minval=0.01, maxval=0.2)
        else:
            print(f"Initializing with values from previous stage (warm start). Target N: {self.hypers.num_intervals}")
            # Ensure initial_f_values is of the correct size for the current stage's num_intervals
            if len(initial_f_values) != self.hypers.num_intervals:
                # Interpolate if resolution changed (using jnp.interp for JAX compatibility)
                print(f"Interpolating f_values from {len(initial_f_values)} to {self.hypers.num_intervals} points.")
                
                # Define x coordinates for interpolation. Assuming f_values represents values over N intervals on [0, 1)
                # So the x-coordinates for initial_f_values would be i * dx_old
                dx_old = 1.0 / len(initial_f_values)
                old_x = jnp.linspace(0, 1.0 - dx_old, len(initial_f_values), endpoint=True) # Represents start of each interval
                
                dx_new = 1.0 / self.hypers.num_intervals
                new_x = jnp.linspace(0, 1.0 - dx_new, self.hypers.num_intervals, endpoint=True)
                
                # jnp.interp handles extrapolation by clamping to min/max if new_x goes outside old_x range.
                f_values = jnp.interp(new_x, old_x, initial_f_values)
                f_values = jax.nn.relu(f_values) # Ensure non-negativity after interpolation
            else:
                f_values = initial_f_values
        
        opt_state = self.optimizer.init(f_values)
        print(f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}, LR: {self.hypers.learning_rate}") # Added LR to print
        train_step_jit = jax.jit(self.train_step)

        best_c2 = -jnp.inf # Track the best C2 value found (inspired by common ML practice)
        best_f_values = f_values # Store the f_values that achieved the best C2
        
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss, c2_val = train_step_jit(f_values, opt_state) # Get c2_val here
            
            # Track the best C2 value found so far
            if c2_val > best_c2:
                best_c2 = c2_val
                best_f_values = f_values # Store the f_values that achieved the best C2

            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:6d} | C2 ≈ {c2_val:.8f} | Best C2: {best_c2:.8f} | Loss: {loss:.4f}") # Log c2_val and best_c2
        
        # Recalculate final C2 from the best f_values to ensure reported C2 is clean, without regularization effect
        _, final_c2 = self._objective_fn(best_f_values)
        print(f"--- Stage '{self.hypers.stage}' Finished ---")
        print(f"Final C2 lower bound for this stage (from best f_values): {final_c2:.8f}")
        return jax.nn.relu(best_f_values), final_c2 # Return the optimized f_values that produced the best C2

def run():
    """Entry point for running the multi-stage optimization (inspired by IP3)."""
    # Stage 1: Exploration - Find a promising function shape from a random start
    hypers_stage1 = Hyperparameters(
        stage="Exploration",
        num_intervals=1024, # Power of 2, good for FFT. Moderate resolution.
        learning_rate=0.003, # Higher LR for exploration
        num_steps=150000, # More steps for thorough exploration
        warmup_steps=15000,
        tv_reg_coeff=1e-6, # Low regularization to encourage some smoothness but not over-constrain
        l2_reg_coeff=1e-7
    )
    optimizer_stage1 = C2Optimizer(hypers_stage1)
    # optimized_f_stage1 will be the f_values
    optimized_f_stage1, _ = optimizer_stage1.run_optimization()

    # Stage 2: Refinement - Fine-tune the result from Stage 1 with higher resolution and smaller learning rate
    hypers_stage2 = Hyperparameters(
        stage="Refinement",
        num_intervals=2048, # Significantly increased resolution for refinement
        learning_rate=5e-5, # Reduced LR for fine-tuning
        num_steps=100000,   # More steps for higher resolution
        warmup_steps=10000,
        tv_reg_coeff=5e-7, # Even lower regularization for fine-tuning
        l2_reg_coeff=5e-8
    )
    optimizer_stage2 = C2Optimizer(hypers_stage2)
    # Use the result of stage 1 as the starting point (warm start), interpolating if N changed
    optimized_f_final, final_c2_val = optimizer_stage2.run_optimization(initial_f_values=optimized_f_stage1)
    
    # Calculate the final loss value including regularization for the final optimized function
    # This loss is what the optimizer actually minimized
    final_objective_value, _ = optimizer_stage2._objective_fn(optimized_f_final)
    loss_val = float(final_objective_value)
    
    f_values_np = np.array(optimized_f_final) # Convert to numpy array for return
    
    # Return the final results and the hyperparameters from the SECOND stage for consistency in reporting n_points
    # The reported n_points should reflect the resolution of the final optimized function.
    return f_values_np, float(final_c2_val), loss_val, hypers_stage2.num_intervals
# EVOLVE-BLOCK-END

