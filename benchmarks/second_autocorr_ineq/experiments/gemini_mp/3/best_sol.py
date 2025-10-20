# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

# Enable float64 for higher precision in numerical calculations
jax.config.update("jax_enable_x64", True)

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    num_intervals: int = 4096 # Further increased resolution for f(x) for higher fidelity. Power of 2 is FFT-friendly.
    learning_rate: float = 0.01 # Retained peak learning rate for strong updates.
    num_steps: int = 400000 # Doubled steps for even more comprehensive exploration in the higher-dimensional space.
    warmup_steps: int = 32000 # Proportional increase in warmup steps.

class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_params: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using the unitless norm calculation.
        f_params are the logarithms of the function values, so f = exp(f_params).
        This ensures f is always positive and avoids zero gradients from relu.
        """
        # Using exp parameterization for guaranteed positivity and smoother gradients
        f_non_negative = jnp.exp(f_params)
        
        N = self.hypers.num_intervals
        # Assume f is discretized over [0, 1], so dx is the spacing.
        # The convolution f*f will be over [0, 2], with the same spacing dx.
        dx = 1.0 / N

        # Calculate the integral of f using the trapezoidal rule for better accuracy.
        # ||f*f||_1 = (∫f)^2
        # This is generally more accurate than a simple rectangular sum.
        integral_f = (jnp.sum(f_non_negative) - 0.5 * (f_non_negative[0] + f_non_negative[-1])) * dx
        # Ensure integral_f is sufficiently positive to avoid division by zero
        integral_f = jnp.maximum(integral_f, 1e-9) 

        # Compute the unscaled discrete autoconvolution using FFT.
        # Padded to 2N for efficient FFT computation.
        # The result (convolution_unscaled) has length 2N.
        padded_f = jnp.pad(f_non_negative, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution_unscaled = jnp.fft.ifft(fft_f * fft_f).real

        # Scale the discrete convolution to represent samples of the continuous (f*f)(x)
        # The continuous convolution is approximated by (sum_k f_k f_{j-k}) * dx
        g_points = convolution_unscaled * dx
        
        # Ensure g_points are non-negative, as f >= 0 implies f*f >= 0
        g_points_non_negative = jax.nn.relu(g_points)

        # Calculate L2-norm squared of the convolution: ||g||₂² = ∫g(x)² dx
        # Using the piecewise-linear integral method (similar to Simpson's rule for g(x)^2)
        # The domain of g is [0, 2]. The spacing for integration is dx.
        # Corrected L2-norm squared integration:
        # The domain of g is [0, 2]. g_points_non_negative has 2N samples (g(0) to g(2-dx)).
        # We need to include g(2), which should be 0 for compactly supported f.
        # This creates 2N intervals over [0, 2], ensuring correct integration range.
        y_points = jnp.concatenate([g_points_non_negative, jnp.array([0.0])])
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((dx / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate L1-norm of the convolution: ||g||₁ = (∫f)²
        norm_1 = integral_f**2
        # Ensure norm_1 is sufficiently positive to avoid division by zero
        norm_1 = jnp.maximum(norm_1, 1e-12)

        # Calculate infinity-norm of the convolution: ||g||_{∞} = sup|g(x)|
        norm_inf = jnp.max(g_points_non_negative)
        # Ensure norm_inf is sufficiently positive
        norm_inf = jnp.maximum(norm_inf, 1e-9)
        
        # Calculate C2 ratio
        denominator = norm_1 * norm_inf
        c2_ratio = l2_norm_squared / denominator
        
        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2_ratio

    def train_step(self, f_params: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_params)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_params)
        f_params = optax.apply_updates(f_params, updates)
        return f_params, opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full optimization process."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-4 # Lower end value for finer convergence in later stages.
        )
        self.optimizer = optax.adam(learning_rate=schedule)
        
        key = jax.random.PRNGKey(42)
        # Initialize log-parameters from a normal distribution.
        # This provides a noisy, non-constant starting point for f, which can aid exploration.
        f_params = jax.random.normal(key, (self.hypers.num_intervals,))
        
        opt_state = self.optimizer.init(f_params)
        print(f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}")
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_params, opt_state, loss = train_step_jit(f_params, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.8f}")
        
        final_c2 = -self._objective_fn(f_params)
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        # Return the actual function values, not the log-parameters.
        return jnp.exp(f_params), final_c2

def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()
    
    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)
    
    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals
# EVOLVE-BLOCK-END

