# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    num_intervals: int = 256 # Increased resolution for f(x)
    learning_rate: float = 0.005 # Adjusted learning rate
    num_steps: int = 50000 # Increased training steps
    warmup_steps: int = 2500 # Adjusted warmup steps

class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant.
    Assumes f is a piecewise constant function on [0, 1], and f(x)=0 outside.
    """
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function (negative C2 ratio) based on the problem definition.
        C₂ = ||f ★ f||₂² / ((∫f)² ||f ★ f||_{∞})
        """
        f_non_negative = jax.nn.relu(f_values)
        
        N = self.hypers.num_intervals
        # dx is the step size for f, assuming f is defined on [0, 1].
        # f_values are samples at 0, dx, ..., (N-1)*dx.
        dx = 1.0 / N 
        
        # 1. Calculate (∫f)² for the denominator.
        # Using Riemann sum for piecewise constant f(x) on [0, 1].
        integral_f = jnp.sum(f_non_negative) * dx
        integral_f_squared = integral_f**2

        # Unscaled discrete autoconvolution
        # Padded to 2N points for linear convolution of N points.
        # The convolution f★f will be defined on [0, 2] with 2N points, step size dx.
        padded_f = jnp.pad(f_non_negative, (0, N)) # Pads to 2N points
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # 2. Calculate ||f ★ f||₂² (∫(f★f)² dx)
        # If f is piecewise constant, f★f is piecewise linear.
        # The formula (h/3)*(y1^2 + y1*y2 + y2^2) is exact for integrating y(x)^2 where y(x) is linear.
        # We apply this over the 2N-1 intervals defined by the 2N convolution points.
        # The total integration domain is effectively [0, (2N-1)*dx] = [0, 2 - 1/N].
        y1_conv_sq = convolution[:-1]**2
        y2_conv_sq = convolution[1:]**2
        y1_y2_conv = convolution[:-1] * convolution[1:]
        
        l2_norm_squared = jnp.sum((dx / 3) * (y1_conv_sq + y1_y2_conv + y2_conv_sq))

        # 3. Calculate ||f ★ f||_{∞}
        # Since f(x) >= 0, f★f(x) >= 0, so max is sufficient.
        norm_inf = jnp.max(convolution)
        
        # Calculate C2 ratio
        denominator = integral_f_squared * norm_inf
        
        # Handle potential division by zero or very small numbers.
        # If f_values are all zero, integral_f will be zero, leading to a zero denominator.
        # We return 0.0 in such cases to encourage non-trivial solutions.
        c2_ratio = jnp.where(denominator > 1e-12, l2_norm_squared / denominator, 0.0)
        
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
        # Initialize f_values uniformly, ensuring positivity.
        f_values = jax.random.uniform(key, (self.hypers.num_intervals,))
        
        opt_state = self.optimizer.init(f_values)
        print(f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}")
        train_step_jit = jax.jit(self.train_step)

        best_c2 = -jnp.inf # Track the best C2 found
        best_f_values = None # Store f_values corresponding to the best C2

        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            current_c2 = -loss
            
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_f_values = f_values # Update best f_values
            
            if step % (self.hypers.num_steps // 10) == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | Current C2 ≈ {current_c2:.8f} | Best C2 ≈ {best_c2:.8f}")
        
        # Re-evaluate C2 with the best found f_values to ensure the reported C2 is accurate
        final_c2 = -self._objective_fn(best_f_values)
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        # Apply relu one last time for the final function output
        return jax.nn.relu(best_f_values), final_c2

def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()
    
    loss_val = -final_c2_val # loss is negative C2
    f_values_np = np.array(optimized_f)
    
    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals
# EVOLVE-BLOCK-END

