# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    num_intervals: int = 200  # Increased for higher resolution, matching world record attempts
    learning_rate: float = 0.01
    num_steps: int = 30000  # Increased steps for convergence with higher N
    warmup_steps: int = 1000
    precision: jnp.dtype = jnp.float64 # Use float64 for higher precision
    regularization_coeff: float = 1e-7 # Small regularization to prevent large f values

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
        Assumes f is a piecewise constant function on [0, 1].
        """
        f_non_negative = jax.nn.relu(f_values)
        
        N = self.hypers.num_intervals
        h_f = 1.0 / N  # Interval width for f, assuming f is on [0, 1]

        # 1. Calculate (integral_f)^2 for the denominator
        # integral_f = integral of f(x) dx over [0, 1]
        # For piecewise constant f, integral_f = sum(f_values) * h_f
        integral_f = jnp.sum(f_non_negative) * h_f
        integral_f_squared = integral_f**2

        # 2. Compute the autoconvolution (f * f)(x)
        # Pad f to 2N elements for FFT convolution. f_non_negative has N points.
        # This implies f is non-zero on [0,1] and zero on (1,2).
        padded_f = jnp.pad(f_non_negative, (0, N)) # Length 2N
        fft_f = jnp.fft.fft(padded_f)
        convolution_unscaled = jnp.fft.ifft(fft_f * fft_f).real

        # Crucial scaling: Discrete convolution result approximates continuous convolution / h_f
        # So, to get samples of (f * f)(x), we multiply by h_f.
        convolution = convolution_unscaled * h_f

        # 3. Calculate L2-norm squared of the convolution (rigorous method)
        # (f * f)(x) is defined on [0, 2] when f is on [0, 1].
        # Convolution has num_conv_points = 2N points, representing samples on [0, 2 - h_conv_sampling].
        # Convolution values C_k = (f*f)(k*h_f) for k=0 to 2N-1, where h_f = 1.0/N.
        # The domain of (f*f)(x) is [0, 2]. We need to integrate (f*f)(x)^2 over [0, 2].
        # We assume (f*f)(2) = 0, which is true for functions supported on [0,1].
        
        # y_points for integration: C_0, C_1, ..., C_{2N-1}, C_{2N}=0.
        # This provides 2N+1 points, defining 2N intervals.
        y_points_for_integral = jnp.concatenate([convolution, jnp.array([0.0], dtype=convolution.dtype)])
        
        # The step size for the convolution samples is h_f = 1.0/N.
        # Total length covered by 2N intervals of width h_f is 2N * (1/N) = 2.0.
        h_conv_for_integral = h_f
        
        # Apply the exact integral for piecewise linear functions (convolution of step functions)
        y1, y2 = y_points_for_integral[:-1], y_points_for_integral[1:]
        l2_norm_squared = jnp.sum((h_conv_for_integral / 3) * (y1**2 + y1 * y2 + y2**2))

        # 4. Calculate infinity-norm of the convolution
        norm_inf = jnp.max(convolution)
        
        # 5. Calculate C2 ratio
        # Denominator uses (integral_f)^2 as per problem definition
        denominator = integral_f_squared * norm_inf
        
        # Handle potential division by zero if integral_f_squared or norm_inf is zero
        # This typically means f_values are all zero, which is not a valid solution.
        c2_ratio = jnp.where(denominator > 1e-12, l2_norm_squared / denominator, 0.0)
        
        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        # Add a small L2 regularization on f_values to prevent them from growing too large.
        # The C2 constant is scale-invariant, so large f values are unnecessary and can cause numerical issues.
        regularization_term = self.hypers.regularization_coeff * jnp.mean(f_non_negative**2)
        return -c2_ratio + regularization_term

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
        # Initialize f_values with the specified precision
        f_values = jax.random.uniform(key, (self.hypers.num_intervals,), dtype=self.hypers.precision)
        
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

