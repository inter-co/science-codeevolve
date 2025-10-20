# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

# Ensure JAX uses float64 for higher precision, crucial for numerical stability in constant discovery
jax.config.update("jax_enable_x64", True)

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    num_intervals: int = 2048 # Doubled resolution for potentially finer details and higher C2, critical for breaking records
    learning_rate: float = 0.00075 # Slightly reduced learning rate for stability with higher N and longer training
    num_steps: int = 300000 # Increased steps for higher resolution and better convergence over extended training
    warmup_steps: int = 15000 # Proportionally increased warmup steps for the longer training schedule
    # Parameters for early stopping, adjusted for longer training and aiming for ultimate precision
    tolerance: float = 5e-10 # Stricter tolerance for C2 improvement, aiming for ultimate precision and stability
    patience: int = 12000 # Increased patience to allow more exploration before early stopping, crucial for fine-tuning

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
        Assumes f is piecewise linear, consistent with convolution norm calculation.
        """
        f_non_negative = jax.nn.relu(f_values)
        
        # Unscaled discrete autoconvolution
        N = self.hypers.num_intervals
        
        # Calculate integral of f (S = ∫f) using Trapezoidal rule for piecewise linear f.
        # This assumes f_non_negative are point samples over [0, 0.5].
        # h_f is the spacing between points.
        if N > 1:
            h_f = 0.5 / (N - 1) 
            integral_f = jnp.sum((f_non_negative[:-1] + f_non_negative[1:]) / 2) * h_f
        else: # Handle N=1 case, though unlikely for high N
            h_f = 0.5
            integral_f = f_non_negative[0] * h_f # Riemann sum for single point
        
        integral_f_squared = integral_f ** 2

        # Unscaled discrete autoconvolution using FFT
        # Padded to 2N for linear convolution.
        padded_f = jnp.pad(f_non_negative, (0, N)) # length 2N
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real # length 2N
        
        # The convolution `convolution` has `num_conv_points = 2N` samples.
        # If f is on [0, 0.5], then f*f is on [0, 1].
        # So these 2N points span the interval [0, 1].
        # For Simpson's rule-like integration, if M points span [a, b], the interval length is (b-a).
        # The spacing `h_conv` is (b-a) / (M-1).
        num_conv_points = len(convolution) # This is 2N
        if num_conv_points > 1:
            h_conv = 1.0 / (num_conv_points - 1)
        else: # Handle num_conv_points = 1 case
            h_conv = 1.0 # If only one point, it spans the whole interval [0,1]
            
        # Calculate L2-norm squared of the convolution.
        # We integrate (f*f)(x)^2 using the trapezoidal rule on the sampled convolution values.
        # This is a standard and robust numerical integration method for sampled data,
        # treating `convolution**2` as a piecewise linear function for integration.
        l2_norm_squared = jax.scipy.integrate.trapezoid(convolution**2, dx=h_conv)

        # Calculate infinity-norm of the convolution
        # Since f_non_negative >= 0, convolution will also be >= 0. So abs is not strictly needed.
        norm_inf = jnp.max(convolution) 
        
        # Calculate C2 ratio: ||f ★ f||₂² / ((∫f)² ||f ★ f||_{∞})
        # Add a small epsilon for numerical stability to avoid division by zero
        epsilon = jnp.array(1e-12, dtype=jnp.float64) # Use float64 epsilon
        denominator = (integral_f_squared + epsilon) * (norm_inf + epsilon)
        
        c2_ratio = l2_norm_squared / denominator
        
        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
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
            end_value=self.hypers.learning_rate * 1e-6 # Smaller end_value for float64 and longer training
        )
        self.optimizer = optax.adam(learning_rate=schedule)
        
        key = jax.random.PRNGKey(42)
        # Initialize f_values with a bell-shaped function (e.g., triangle) to provide a structured starting point.
        # This can help the optimizer converge faster to a good local optimum compared to purely random initialization.
        x_points = jnp.linspace(0.0, 0.5, self.hypers.num_intervals, dtype=jnp.float64)
        # A simple triangle function, scaled to have a peak height around 0.1-0.2.
        # The optimal function often has a bell-like shape, making this a good prior.
        f_initial_shape = 0.2 * (1.0 - jnp.abs(x_points - 0.25) / 0.25)
        f_initial_shape = jnp.maximum(0.01, f_initial_shape) # Ensure strictly positive, with a minimum value to avoid zero gradients.
        
        # Add a small amount of noise to break perfect symmetry and allow exploration around the initial guess.
        noise = jax.random.uniform(key, (self.hypers.num_intervals,), dtype=jnp.float64) * 0.005
        f_values = f_initial_shape + noise
        
        opt_state = self.optimizer.init(f_values)
        print(f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}")
        train_step_jit = jax.jit(self.train_step)

        # Initialize variables for tracking best C2 and early stopping
        best_c2 = -jnp.inf
        best_f_values = None
        patience_counter = 0

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            current_c2 = -loss # C2 is the negative of the loss

            # Track best C2 and apply early stopping
            if current_c2 > best_c2 + self.hypers.tolerance:
                best_c2 = current_c2
                best_f_values = f_values
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress periodically or at the last step
            if step % (self.hypers.num_steps // 10) == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | Current C2 ≈ {current_c2:.12f} | Best C2 ≈ {best_c2:.12f} | Patience: {patience_counter}")

            # Check for early stopping condition
            if patience_counter >= self.hypers.patience:
                print(f"Early stopping at step {step} due to lack of improvement.")
                break
        
        # Use the best f_values found during training, or the final if no improvement was found
        final_f_values = best_f_values if best_f_values is not None else f_values
        # Recalculate C2 for the best function to ensure consistency
        final_c2 = -self._objective_fn(final_f_values) 
        
        print(f"Optimization finished. Final C2 lower bound found: {final_c2:.12f}")
        return jax.nn.relu(final_f_values), final_c2

def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()
    
    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)
    
    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals
# EVOLVE-BLOCK-END

