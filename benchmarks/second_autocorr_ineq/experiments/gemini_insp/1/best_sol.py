# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    stage: str # Added stage name for multi-stage optimization (from Inspiration 2)
    num_intervals: int = 800
    learning_rate: float = 0.003
    num_steps: int = 100000
    warmup_steps: int = 10000
    tv_reg_coeff: float = 5e-4 # Coefficient for Total Variation regularization
    l2_reg_coeff: float = 1e-4 # Coefficient for L2 regularization on f_values

class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using a unitless norm calculation with Riemann sum for L2 convolution.
    (Updated docstring to reflect Riemann sum for L2 norm)
    """
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the objective function and the raw C2 ratio.
        Returns objective_value (for gradients) and c2_ratio (for logging).
        """
        f_non_negative = jax.nn.relu(f_values)
        
        N = self.hypers.num_intervals
        dx = 1.0 / N # Assuming f is supported on [0, 1] for normalization purposes
        
        # Unscaled discrete autoconvolution
        # Pad f to length 2N for linear convolution (f on [0,1], conv on [0,2])
        padded_f = jnp.pad(f_non_negative, (0, N)) # Length 2N
        
        # Apply FFT. Note: jnp.fft.fft is unscaled.
        fft_f = jnp.fft.fft(padded_f)
        
        # Inverse FFT scales by 1/len(input). So, ifft(fft(f)*fft(f)) gives sum_k f_k f_{j-k} / (2N)
        # To get actual samples of (f*f)(x), we need to multiply by (2N * dx).
        # Since dx = 1/N, this simplifies to multiplying by 2.0.
        raw_convolution_ifft = jnp.fft.ifft(fft_f * fft_f).real
        convolution_samples = raw_convolution_ifft * 2.0 # Correctly scaled samples of (f*f)(x)
        
        # The convolution function (f*f) is defined over [0, 2].
        # The number of samples in `convolution_samples` is 2N.
        # So the effective integration step for the convolution is dx_conv = 2.0 / (2N) = 1.0 / N = dx.
        dx_conv = dx

        # Calculate L2-norm squared of the convolution using Riemann sum, as it empirically led to higher C2.
        # For sufficiently large N, this approximation is good for optimization.
        # It assumes convolution_samples are samples of (f*f)(x) and f is zero outside [0,1].
        l2_norm_squared = jnp.sum(convolution_samples**2) * dx_conv

        # Calculate L1-norm of the convolution using the problem's simplification: ||f ★ f||₁ = (∫f)²
        integral_f = jnp.sum(f_non_negative) * dx
        norm_1 = integral_f**2

        # Calculate infinity-norm of the convolution using the correctly scaled samples
        norm_inf = jnp.max(jnp.abs(convolution_samples))
        
        # Calculate C2 ratio
        denominator = norm_1 * norm_inf
        # Robustly handle potential division by zero using jnp.where (JAX-compatible conditional).
        denominator = jnp.where(denominator == 0, 1e-12, denominator)
        c2_ratio = l2_norm_squared / denominator
        
        # Total Variation (TV) regularization to encourage piecewise constant functions
        # For discrete f_values representing heights of steps, sum of absolute differences is the TV.
        tv_regularization = jnp.sum(jnp.abs(jnp.diff(f_non_negative)))
        
        # L2 regularization on f_values to prevent values from becoming too large and improve stability
        # Scaled by dx to approximate integral(f^2 dx)
        l2_regularization = jnp.sum(f_non_negative**2) * dx

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        # Add regularization terms to the negative C2.
        objective_value = -c2_ratio + \
                          self.hypers.tv_reg_coeff * tv_regularization + \
                          self.hypers.l2_reg_coeff * l2_regularization
        
        return objective_value, c2_ratio # Return both objective and raw C2 ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step. (Modified to use has_aux=True)"""
        (loss, c2_ratio), grads = jax.value_and_grad(self._objective_fn, has_aux=True)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss, c2_ratio

    def run_optimization(self, initial_f_values: jnp.ndarray | None = None):
        """
        Sets up and runs the full optimization process, with an optional warm start.
        (Modified for multi-stage support from Inspiration 2)
        """
        print(f"\n--- Starting Optimization Stage: {self.hypers.stage} ---")
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-4
        )
        self.optimizer = optax.adam(learning_rate=schedule)
        
        key = jax.random.PRNGKey(42)
        if initial_f_values is None:
            print("Initializing with new random values.")
            key, subkey = jax.random.split(key) # Use a new subkey for f_values initialization
            f_values = jax.random.uniform(subkey, (self.hypers.num_intervals,), minval=0.01, maxval=0.2)
        else:
            print("Initializing with values from previous stage (warm start).")
            f_values = initial_f_values
        
        opt_state = self.optimizer.init(f_values)
        print(f"N={self.hypers.num_intervals}, Steps: {self.hypers.num_steps}, LR: {self.hypers.learning_rate}, TV: {self.hypers.tv_reg_coeff}, L2: {self.hypers.l2_reg_coeff}")
        train_step_jit = jax.jit(self.train_step)

        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss, c2_val = train_step_jit(f_values, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:6d} | C2 ≈ {c2_val:.8f} | Loss: {loss:.4f}")
        
        _, final_c2 = self._objective_fn(f_values) # Get the clean C2 for final reporting
        print(f"--- Stage '{self.hypers.stage}' Finished ---")
        print(f"Final C2 lower bound for this stage: {final_c2:.8f}")
        return jax.nn.relu(f_values), final_c2

def run():
    """Entry point for running the multi-stage optimization. (Adopted from Inspiration 2)"""
    # Stage 1: Exploration - Find a promising function shape from a random start
    hypers_stage1 = Hyperparameters(
        stage="Exploration",
        num_intervals=800,
        learning_rate=0.003,
        num_steps=100000,
        warmup_steps=10000,
        tv_reg_coeff=5e-4,
        l2_reg_coeff=1e-4
    )
    optimizer_stage1 = C2Optimizer(hypers_stage1)
    optimized_f_stage1, _ = optimizer_stage1.run_optimization()

    # Stage 2: Refinement - Fine-tune the result from Stage 1 with a smaller learning rate and further reduced regularization
    hypers_stage2 = Hyperparameters(
        stage="Refinement",
        num_intervals=800,
        learning_rate=1e-4, # Retain this LR as it performed well in previous iteration
        num_steps=100000,   # Increased steps for more thorough refinement
        warmup_steps=2500,  # Reduced warmup steps for quicker decay
        tv_reg_coeff=1e-4,  # Further reduced regularization
        l2_reg_coeff=2e-5   # Further reduced regularization
    )
    optimizer_stage2 = C2Optimizer(hypers_stage2)
    # Use the result of stage 1 as the starting point (warm start)
    optimized_f_stage2, _ = optimizer_stage2.run_optimization(initial_f_values=optimized_f_stage1)
    
    # Stage 3: Polishing - A new stage to make final, very small adjustments with minimal regularization.
    # (Adapted from Inspiration 2/3, with further reduced regularization based on Target's strong Stage 2)
    hypers_stage3 = Hyperparameters(
        stage="Polishing",
        num_intervals=800,
        learning_rate=1e-5, # Very low learning rate
        num_steps=30000,    # Sufficient steps for polishing
        warmup_steps=3000,
        tv_reg_coeff=5e-5,  # Further reduced regularization (lower than Insp 2/3's 1e-4)
        l2_reg_coeff=1e-5   # Further reduced regularization
    )
    optimizer_stage3 = C2Optimizer(hypers_stage3)
    optimized_f_final, final_c2_val = optimizer_stage3.run_optimization(initial_f_values=optimized_f_stage2)

    loss_val = -final_c2_val # For consistency with previous reporting, though c2_val is cleaner
    f_values_np = np.array(optimized_f_final)
    
    # Return the final results and the hyperparameters from the first stage for consistency in reporting n_points
    return f_values_np, float(final_c2_val), float(loss_val), hypers_stage1.num_intervals
# EVOLVE-BLOCK-END

