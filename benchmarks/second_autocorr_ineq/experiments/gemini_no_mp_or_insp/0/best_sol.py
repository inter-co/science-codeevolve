# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    num_intervals: int = 512 # Doubled resolution for better function representation, crucial for higher C2
    learning_rate: float = 0.005 # Adjusted learning rate, often smaller for higher N and longer training
    num_steps: int = 100000 # More steps for higher resolution and better convergence (doubled for 2x intervals)
    warmup_steps: int = 5000 # Proportionally increased warmup steps (doubled)
    # Added parameters for early stopping to improve efficiency and stability
    tolerance: float = 1e-8 # Minimum change in C2 to be considered an improvement (stricter for finer convergence)
    patience: int = 4000 # Number of steps to wait for improvement before stopping (doubled for higher N)

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
        """
        f_non_negative = jax.nn.relu(f_values)
        
        # Unscaled discrete autoconvolution
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # Assume f is defined on [0, L_f].
        # The problem statement's C2 formula uses (∫f)².
        # The original code's L2/L1 norms implicitly integrate f*f over an interval of length 1.
        # If convolution g = f*f is implicitly scaled to be on [0, 1],
        # then the domain of f must be [0, 0.5] (since f*f's domain is 2*f's domain).
        # h_f is the spacing for samples of f on [0, 0.5].
        h_f = 0.5 / N 

        # Calculate integral of f (S = ∫f) using Riemann sum approximation for piecewise constant f.
        integral_f = jnp.sum(f_non_negative) * h_f
        integral_f_squared = integral_f ** 2

        # Unscaled discrete autoconvolution
        # Padded to 2N for linear convolution.
        padded_f = jnp.pad(f_non_negative, (0, N)) # length 2N
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real # length 2N

        # The original code's h for norms (1.0 / (len(convolution) + 1)) implies
        # integration over an interval of length 1 for the convolution.
        # We will keep this `h_conv` for consistency with the original L2 norm calculation,
        # which performs integration using piecewise linear approximation (Simpson's rule for g^2).
        num_conv_points = len(convolution) # This is 2N
        h_conv = 1.0 / (num_conv_points + 1) # This is 1.0 / (2N + 1)

        # Calculate L2-norm squared of the convolution (rigorous method, piecewise linear approx)
        # The y_points construction adds 0.0 at start/end, making 2N+2 points for 2N+1 intervals.
        # This is consistent with integrating over a length 1 interval using h_conv.
        y_points = jnp.concatenate([jnp.array([0.0]), convolution, jnp.array([0.0])])
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((h_conv / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate infinity-norm of the convolution
        norm_inf = jnp.max(jnp.abs(convolution))
        
        # Calculate C2 ratio: ||f ★ f||₂² / ((∫f)² ||f ★ f||_{∞})
        # Add a small epsilon for numerical stability to avoid division by zero
        epsilon = 1e-10
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
            end_value=self.hypers.learning_rate * 1e-4
        )
        self.optimizer = optax.adam(learning_rate=schedule)
        
        key = jax.random.PRNGKey(42)
        # Initialize f_values with a simple box function, which is a good starting point for step functions.
        # This can help the optimizer converge faster to a better local optimum, especially given the
        # current world record is achieved by step functions.
        f_values = jnp.zeros(self.hypers.num_intervals)
        h_f_initial = 0.5 / self.hypers.num_intervals
        # Define a central box, for example from 0.1 to 0.4 of the domain [0, 0.5]
        # This translates to indices for f_values
        start_idx = int(0.1 / h_f_initial)
        end_idx = int(0.4 / h_f_initial)
        # Ensure indices are within bounds
        start_idx = jnp.clip(start_idx, 0, self.hypers.num_intervals - 1)
        end_idx = jnp.clip(end_idx, 0, self.hypers.num_intervals - 1)
        # Set values to 1.0 in the box region. Using `at[...].set()` for JAX immutability.
        f_values = f_values.at[start_idx:end_idx+1].set(1.0)
        # Add a small amount of uniform noise to break symmetry and aid exploration
        f_values = f_values + jax.random.uniform(key, (self.hypers.num_intervals,)) * 0.01
        
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
                print(f"Step {step:5d} | Current C2 ≈ {current_c2:.8f} | Best C2 ≈ {best_c2:.8f} | Patience: {patience_counter}")

            # Check for early stopping condition
            if patience_counter >= self.hypers.patience:
                print(f"Early stopping at step {step} due to lack of improvement.")
                break
        
        # Use the best f_values found during training, or the final if no improvement was found
        final_f_values = best_f_values if best_f_values is not None else f_values
        # Recalculate C2 for the best function to ensure consistency
        final_c2 = -self._objective_fn(final_f_values) 
        
        print(f"Optimization finished. Final C2 lower bound found: {final_c2:.8f}")
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

