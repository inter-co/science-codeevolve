# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    num_intervals: int = 1024 # Doubled resolution for finer function details, crucial for complex shapes
    learning_rate: float = 0.005 # Reduced learning rate for stability with higher resolution and more steps
    num_steps: int = 250000 # Significantly increased training steps for better convergence at higher resolution
    warmup_steps: int = 25000 # Adjusted warmup steps (10% of num_steps)
    num_restarts: int = 8 # Increased restarts to explore the more complex, higher-dimensional landscape

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
        This version directly uses (∫f)^2 in the denominator for mathematical consistency.
        """
        f_non_negative = jax.nn.relu(f_values)
        
        N = self.hypers.num_intervals
        
        # Calculate integral_f for the denominator: (∫f)²
        # Assume f_values are samples of f on [0, 1], so dx_f = 1/N.
        # Using rectangular rule for ∫f.
        integral_f = jnp.sum(f_non_negative) / N
        integral_f_squared = integral_f**2
        
        # Ensure integral_f_squared is not zero to prevent division by zero.
        # Add a small epsilon for numerical stability.
        integral_f_squared = jnp.maximum(integral_f_squared, 1e-10)

        # Unscaled discrete autoconvolution
        # Pad f to 2N length. If f is on [0,1], its convolution f*f is on [0,2].
        padded_f = jnp.pad(f_non_negative, (0, N)) # Length 2N
        fft_f = jnp.fft.fft(padded_f)
        convolution_result = jnp.fft.ifft(fft_f * fft_f).real # Length 2N

        # Determine the effective step size for convolution's domain [0, 2].
        # If f is sampled on [0,1] with N points, convolution is on [0,2] with 2N points.
        # Thus, dx_conv = 2.0 / len(convolution_result) = 2.0 / (2N) = 1.0 / N.
        dx_conv = 1.0 / N

        # For L2-norm squared of the convolution (∫(f*f)^2 dx),
        # we consider f*f to be a piecewise linear function.
        # The `convolution_result` contains `2N` samples from (f*f)(0) to (f*f)(2 - dx_conv).
        # We need to append (f*f)(2) which is 0 for compactly supported f.
        # This creates `2N+1` points for `2N` intervals, covering the domain [0, 2].
        g_points = jnp.concatenate([convolution_result, jnp.array([0.0])])
        
        # This formula is for integrating the square of a piecewise linear function.
        # Here, g1 and g2 are the values of (f*f) at the endpoints of each interval.
        g1, g2 = g_points[:-1], g_points[1:]
        l2_norm_squared = jnp.sum((dx_conv / 3) * (g1**2 + g1 * g2 + g2**2))

        # Calculate infinity-norm of the convolution.
        norm_inf = jnp.max(jnp.abs(convolution_result))
        
        # Ensure norm_inf is not zero to prevent division by zero.
        norm_inf = jnp.maximum(norm_inf, 1e-10)
        
        # Calculate C2 ratio: ||f ★ f||₂² / ((∫f)² ||f ★ f||_{∞})
        denominator = integral_f_squared * norm_inf
        c2_ratio = l2_norm_squared / denominator
        
        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss

    def run_optimization(self, key: jax.random.PRNGKey):
        """Sets up and runs the full optimization process."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-4
        )
        # Combine Adam with gradient clipping for stability, especially with complex loss landscape
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), # Clip gradients by global norm to prevent explosions
            optax.adam(learning_rate=schedule)
        )
        
        # Initialize f_values with random uniform values for more diverse exploration.
        # The relu activation in _objective_fn ensures non-negativity.
        f_values = jax.random.uniform(key, (self.hypers.num_intervals,), minval=0.0, maxval=1.0)
        
        opt_state = self.optimizer.init(f_values)
        print(f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}")
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if step % (self.hypers.num_steps // 10) == 0 or step == self.hypers.num_steps - 1: # Print 10 times during run
                print(f"Step {step:5d} | C2 ≈ {-loss:.8f}")
        
        final_c2 = -self._objective_fn(f_values)
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        return jax.nn.relu(f_values), final_c2

def run():
    """Entry point for running the optimization with multiple restarts."""
    hypers = Hyperparameters()
    
    best_c2_val = -jnp.inf # Initialize with negative infinity
    best_optimized_f = None
    
    base_key = jax.random.PRNGKey(42) # Fixed seed for reproducibility of multi-runs
    
    for i in range(hypers.num_restarts):
        print(f"\n--- Starting Optimization Run {i+1}/{hypers.num_restarts} ---")
        optimizer = C2Optimizer(hypers)
        
        # Split key for each run to ensure different random initializations
        run_key, base_key = jax.random.split(base_key)
        
        optimized_f, current_c2_val = optimizer.run_optimization(run_key)
        
        if current_c2_val > best_c2_val:
            best_c2_val = current_c2_val
            best_optimized_f = optimized_f
            print(f"New best C2 found: {best_c2_val:.8f}")
    
    loss_val = -best_c2_val
    f_values_np = np.array(best_optimized_f)
    
    return f_values_np, float(best_c2_val), float(loss_val), hypers.num_intervals
# EVOLVE-BLOCK-END

