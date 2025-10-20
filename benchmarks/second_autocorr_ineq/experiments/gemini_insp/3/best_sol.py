# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    stage: str = "Default" # Added stage for multi-stage optimization (from Insp 1)
    num_intervals: int = 1024 # Power of 2 for FFT efficiency. Initial resolution for exploration.
    learning_rate: float = 0.003 # Learning rate for initial exploration
    num_steps: int = 150000 # Increased steps for thorough exploration (from Insp 1)
    warmup_steps: int = 15000 # Adjusted warmup steps (scaled from Insp 1)
    tv_reg_coeff: float = 1e-6 # Coefficient for Total Variation regularization (from Insp 1)
    weight_decay: float = 1e-7 # Decoupled weight decay for AdamW (from Insp 1)

class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using a multi-stage approach with regularization and high precision.
    """
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]: # Modified to return both objective and c2_ratio (from Insp 1)
        """
        Computes the objective function using the rigorous, unitless, piecewise-linear integral method.
        f is assumed to be piecewise constant on [0, 1].
        All computations use float64 for higher precision, crucial for record-breaking attempts.
        """
        # Ensure f_values are non-negative and use float64
        f_non_negative = jax.nn.relu(f_values).astype(jnp.float64)
        
        N = self.hypers.num_intervals
        L = 1.0  # Assume domain of f is [0, L]
        dx = L / N

        # 1. Calculate ||f ★ f||_1 = (∫f)^2
        integral_f = jnp.sum(f_non_negative) * dx
        norm_1 = integral_f**2

        # 2. Compute convolution g = f ★ f using FFT
        # Pad f to length 2N for linear convolution.
        # If f is on [0,L], its convolution f*f is on [0,2L].
        # For L=1, f is on [0,1], padded to [0,2] for FFT.
        padded_f = jnp.pad(f_non_negative, (0, N)) # Length 2N
        fft_f = jnp.fft.fft(padded_f)
        
        # Inverse FFT scales by 1/len(input). To get samples of (f*f)(x),
        # we need to multiply by (len(input) * dx_f).
        # Here, len(input) = 2N, dx_f = 1/N (since f is piecewise constant on N intervals of width dx=1/N).
        # So, scaling factor is 2N * (1/N) = 2.0. (Corrected from target program, aligned with Insp 1 & 3)
        raw_convolution_ifft = jnp.fft.ifft(fft_f * fft_f).real # Convolution has length 2N
        convolution_samples = raw_convolution_ifft * 2.0 # Correctly scaled (from Insp 1 & 3)
        
        # Ensure convolution remains non-negative due to potential numerical noise from IFFT (from Insp 1)
        convolution_samples = jax.nn.relu(convolution_samples).astype(jnp.float64) # Ensure float64

        # 3. Calculate ||f ★ f||_infinity = sup |g(x)|
        norm_inf = jnp.max(convolution_samples)
        
        # 4. Calculate ||f ★ f||_2^2 = ∫g(x)^2 dx using Simpson's rule for piecewise linear.
        # The convolution 'g' has 2N samples and spans the domain [0, 2L] = [0, 2].
        # The discretization step for the convolution domain is dx_conv = 2.0 / (2N) = 1.0 / N = dx.
        dx_conv = dx
        
        # For Simpson's rule, we consider the 2N convolution samples as g(0), ..., g((2N-1)dx_conv).
        # We need to add g(2L)=0 (since f has finite support) to define the last interval.
        y_points = jnp.concatenate([convolution_samples, jnp.array([0.0], dtype=jnp.float64)])
        y1, y2 = y_points[:-1], y_points[1:] # Creates 2N pairs for 2N intervals
        l2_norm_squared = jnp.sum((dx_conv / 3) * (y1**2 + y1 * y2 + y2**2)) # Simpson's rule (from target & Insp 2)

        # Calculate C2 ratio
        denominator = norm_1 * norm_inf
        # Add a small epsilon to the denominator for numerical stability. (from Insp 1)
        denominator = jnp.where(denominator < 1e-12, 1e-12, denominator)
        c2_ratio = l2_norm_squared / denominator

        # Total Variation (TV) regularization to encourage piecewise constant functions (from Insp 1)
        tv_regularization = jnp.sum(jnp.abs(jnp.diff(f_non_negative)))
        
        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        # L2 regularization is now handled by the AdamW optimizer's weight_decay.
        objective_value = -c2_ratio + self.hypers.tv_reg_coeff * tv_regularization
        
        return objective_value, c2_ratio # Return both for logging (from Insp 1)

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        # Modified to handle has_aux=True for objective_fn and return c2_val (from Insp 1)
        (loss, c2_val), grads = jax.value_and_grad(self._objective_fn, has_aux=True)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss, c2_val # Return c2_val for logging (from Insp 1)

    def run_optimization(self, initial_f_values: jnp.ndarray | None = None): # Added initial_f_values for warm-starting (from Insp 1)
        """Sets up and runs the full optimization process for a stage."""
        print(f"\n--- Starting Optimization Stage: {self.hypers.stage} ---") # Added stage info (from Insp 1)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-4
        )
        # Evolved to use AdamW with decoupled weight decay for potentially better optimization (from Insp 1).
        self.optimizer = optax.adamw(
            learning_rate=schedule,
            weight_decay=self.hypers.weight_decay
        )
        
        key = jax.random.PRNGKey(42) # Fixed seed for reproducibility
        if initial_f_values is None:
            print("Initializing with new random uniform values for f_values.")
            key, subkey = jax.random.split(key) # Use subkey for f_values
            # Initialize f_values with small random uniform values to encourage diverse exploration (from Insp 1)
            f_values = jax.random.uniform(subkey, (self.hypers.num_intervals,), minval=0.01, maxval=0.2, dtype=jnp.float64)
        else:
            print(f"Initializing with values from previous stage (warm start). Target N: {self.hypers.num_intervals}")
            # Ensure initial_f_values is of the correct size for the current stage's num_intervals
            if len(initial_f_values) != self.hypers.num_intervals:
                # Interpolate if resolution changed (using jnp.interp for JAX compatibility, from Insp 1)
                print(f"Interpolating f_values from {len(initial_f_values)} to {self.hypers.num_intervals} points.")
                
                # Define x coordinates for interpolation. Assuming f_values represents values over N intervals on [0, 1)
                dx_old = 1.0 / len(initial_f_values)
                old_x = jnp.linspace(0, 1.0 - dx_old, len(initial_f_values), endpoint=False, dtype=jnp.float64)
                
                dx_new = 1.0 / self.hypers.num_intervals
                new_x = jnp.linspace(0, 1.0 - dx_new, self.hypers.num_intervals, endpoint=False, dtype=jnp.float64)
                
                # jnp.interp handles extrapolation by clamping to min/max if new_x goes outside old_x range.
                f_values = jnp.interp(new_x, old_x, initial_f_values).astype(jnp.float64)
                f_values = jax.nn.relu(f_values) # Ensure non-negativity after interpolation
            else:
                f_values = initial_f_values.astype(jnp.float64) # Ensure float64 consistency
        
        opt_state = self.optimizer.init(f_values)
        print(f"N: {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}, LR: {self.hypers.learning_rate}, TV: {self.hypers.tv_reg_coeff}, WD: {self.hypers.weight_decay}") # Added more params to print
        train_step_jit = jax.jit(self.train_step)

        best_c2 = -jnp.inf # Track the best C2 value found (from Insp 1)
        best_f_values = f_values # Store the f_values that achieved the best C2
        
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss, c2_val = train_step_jit(f_values, opt_state) # Get c2_val here (from Insp 1)
            
            # Track the best C2 value found so far (from Insp 1)
            if c2_val > best_c2:
                best_c2 = c2_val
                best_f_values = f_values # Store the f_values that achieved the best C2

            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:6d} | C2 ≈ {c2_val:.8f} | Best C2: {best_c2:.8f} | Loss: {loss:.4f}") # Log c2_val and best_c2 (from Insp 1)
        
        # Recalculate final C2 from the best f_values to ensure reported C2 is clean, without regularization effect (from Insp 1)
        _, final_c2 = self._objective_fn(best_f_values)
        print(f"--- Stage '{self.hypers.stage}' Finished ---")
        print(f"Final C2 lower bound for this stage (from best f_values): {final_c2:.8f}")
        return jax.nn.relu(best_f_values), final_c2 # Return the optimized f_values that produced the best C2 (from Insp 1)

def run():
    """Entry point for running the multi-stage optimization (from Insp 1, highest C2)."""
    # Stage 1: Exploration - Find a promising function shape from a random start
    hypers_stage1 = Hyperparameters(
        stage="Exploration",
        num_intervals=1024, # Power of 2, good for FFT. Moderate resolution. (from Insp 1)
        learning_rate=0.003, # Higher LR for exploration (from Insp 1)
        num_steps=150000, # More steps for thorough exploration (from Insp 1)
        warmup_steps=15000, # Adjusted warmup steps proportionally (from Insp 1)
        tv_reg_coeff=1e-6, # Low regularization to encourage some smoothness but not over-constrain (from Insp 1)
        weight_decay=1e-7 # Decoupled weight decay for AdamW (from Insp 1)
    )
    optimizer_stage1 = C2Optimizer(hypers_stage1)
    optimized_f_stage1, _ = optimizer_stage1.run_optimization()

    # Stage 2: Refinement - Fine-tune the result from Stage 1 with higher resolution and slightly increased learning rate
    hypers_stage2 = Hyperparameters(
        stage="Refinement",
        num_intervals=2048, # Significantly increased resolution for refinement (from Insp 1)
        learning_rate=6e-5, # Slightly increased LR for better exploration in this stage (from Insp 1)
        num_steps=150000,   # Matched steps to Stage 1 for thorough refinement (from Insp 1)
        warmup_steps=15000, # Adjusted warmup steps proportionally (from Insp 1)
        tv_reg_coeff=3e-7, # Maintained lower regularization for more complex features (from Insp 1)
        weight_decay=3e-8  # Decoupled weight decay for AdamW (from Insp 1)
    )
    optimizer_stage2 = C2Optimizer(hypers_stage2)
    optimized_f_stage2, _ = optimizer_stage2.run_optimization(initial_f_values=optimized_f_stage1)

    # Stage 3: Ultra-Refinement - Even higher resolution, very fine-tuning
    hypers_stage3 = Hyperparameters(
        stage="Ultra-Refinement",
        num_intervals=4096, # Doubled resolution for ultimate precision (from Insp 1)
        learning_rate=1e-5, # Very low LR for ultra-fine-tuning (from Insp 1)
        num_steps=75000,    # Moderate steps for final adjustments (from Insp 1)
        warmup_steps=7500,  # Adjusted warmup steps proportionally (from Insp 1)
        tv_reg_coeff=1e-7,  # Extremely low regularization to allow maximum complexity (from Insp 1)
        weight_decay=1e-8   # Decoupled weight decay for AdamW (from Insp 1)
    )
    optimizer_stage3 = C2Optimizer(hypers_stage3)
    optimized_f_final, final_c2_val = optimizer_stage3.run_optimization(initial_f_values=optimized_f_stage2)
    
    # Calculate the final loss value including regularization for the final optimized function
    # This loss is what the optimizer actually minimized
    final_objective_value, _ = optimizer_stage3._objective_fn(optimized_f_final)
    loss_val = float(final_objective_value)
    
    f_values_np = np.array(optimized_f_final) # Convert to numpy array for return
    
    # Return the final results and the hyperparameters from the FINAL (THIRD) stage for consistency in reporting n_points (from Insp 1)
    # The reported n_points should reflect the resolution of the final optimized function.
    return f_values_np, float(final_c2_val), loss_val, hypers_stage3.num_intervals
# EVOLVE-BLOCK-END

