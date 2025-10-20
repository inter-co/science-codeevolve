# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    num_intervals: int = 1024 # Increased resolution (power of 2 for FFT efficiency), to potentially capture finer details
    learning_rate: float = 0.003 # Retain learning rate for stability (from Insp 1)
    num_steps: int = 150000 # Increased steps for more thorough optimization with higher resolution
    warmup_steps: int = 15000 # Scaled proportionally to num_steps
    tv_reg_coeff: float = 5e-4 # Coefficient for Total Variation regularization (encourages piecewise constant functions, from Insp 1)
    l2_reg_coeff: float = 1e-4 # Coefficient for L2 regularization on f_values (prevents excessive magnitudes, from Insp 1)
    # Removed symmetry_reg_coeff as it was not present in the highest-performing inspiration programs.

class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using the unitless norm calculation.
        Assumes f_values define a piecewise constant function on [0, 1].
        """
        f_non_negative = jax.nn.relu(f_values) # Ensure f(x) >= 0
        
        N = self.hypers.num_intervals
        dx = 1.0 / N # Discretization step for f on [0, 1]
        
        # Unscaled discrete autoconvolution
        # Pad f to length 2N for linear convolution.
        # f is on [0,1], its convolution f*f is on [0,2].
        padded_f = jnp.pad(f_non_negative, (0, N)) # Length 2N (from Insp 1)
        
        # Apply FFT. jnp.fft.fft is unscaled.
        fft_f = jnp.fft.fft(padded_f)
        
        # Inverse FFT scales by 1/len(input). So, ifft(fft(f)*fft(f)) gives sum_k f_k f_{j-k} / (2N)
        # To get actual samples of (f*f)(x), we need to multiply by (2N * dx_f).
        # Since dx_f = 1/N, this simplifies to multiplying by 2.0. (from Insp 1)
        raw_convolution_ifft = jnp.fft.ifft(fft_f * fft_f).real
        convolution_samples = raw_convolution_ifft * 2.0 # Correctly scaled samples of (f*f)(x) (from Insp 1)
        
        # The convolution function (f*f) is defined over [0, 2].
        # The number of samples in `convolution_samples` is 2N.
        # So the effective integration step for the convolution for Riemann sum is dx_conv = 2.0 / (2N) = 1.0 / N = dx.
        dx_conv = dx

        # Calculate L2-norm squared of the convolution (using Riemann sum, as in Insp 1)
        l2_norm_squared = jnp.sum(convolution_samples**2) * dx_conv # (from Insp 1)

        # Calculate L1-norm of the convolution using the problem's simplification: ||f ★ f||₁ = (∫f)²
        integral_f = jnp.sum(f_non_negative) * dx # Consistent with dx definition
        norm_1 = integral_f**2

        # Calculate infinity-norm of the convolution using the correctly scaled samples (from Insp 1)
        norm_inf = jnp.max(jnp.abs(convolution_samples)) 
        
        # Calculate C2 ratio
        denominator = norm_1 * norm_inf
        # Add a small epsilon to the denominator for numerical stability, to prevent division by zero.
        # Inspiration 1 did not have this, but it's a good practice (adopted from Insp 3 in previous turn).
        c2_ratio = l2_norm_squared / jnp.maximum(denominator, 1e-12)

        # Total Variation (TV) regularization to encourage piecewise constant functions (from Insp 1)
        tv_regularization = jnp.sum(jnp.abs(jnp.diff(f_non_negative)))
        
        # L2 regularization on f_values to prevent values from becoming too large and improve stability (from Insp 1)
        l2_regularization = jnp.sum(f_non_negative**2) * dx

        # Removed: Soft symmetry regularization (it was not present in the highest-performing inspiration programs)
        
        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        # Add regularization terms to the negative C2.
        objective_value = -c2_ratio + \
                          self.hypers.tv_reg_coeff * tv_regularization + \
                          self.hypers.l2_reg_coeff * l2_regularization
        
        return objective_value

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        # The loss here is the objective_value computed in _objective_fn
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full optimization process."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-4
        )
        self.optimizer = optax.adam(learning_rate=schedule)
        
        key = jax.random.PRNGKey(42)
        # Initialize f_values with small random uniform values to encourage diverse exploration
        # and prevent symmetry issues that a constant initialization might introduce.
        # Values are kept positive (minval=0.01) to align with f(x) >= 0 constraint and ensure non-trivial starting points.
        key, subkey = jax.random.split(key) # Use a new subkey for f_values initialization to maintain global PRNGKey state (from Insp 1)
        f_values = jax.random.uniform(subkey, (self.hypers.num_intervals,), minval=0.01, maxval=0.2) # (from Insp 1)
        
        opt_state = self.optimizer.init(f_values)
        print(f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}")
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.8f}")
        
        final_c2 = -self._objective_fn(f_values)
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        return jax.nn.relu(f_values), final_c2

def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()
    
    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)
    
    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals
# EVOLVE-BLOCK-END

