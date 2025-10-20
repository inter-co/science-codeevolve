# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    num_intervals: int = 400 # Massively increase resolution for maximum fidelity
    learning_rate: float = 0.01 # Revert to previously successful, larger learning rate
    num_steps: int = 50000  # Increase steps for convergence in higher-dimensional space
    warmup_steps: int = 5000 # Adjusted to scale with num_steps
    regularization_strength: float = 1e-7 # Strength of the smoothness penalty

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

        # Number of points in the convolution array (length 2N)
        num_conv_points = len(convolution)
        # The convolution of f on [0,1] with itself is on [0,2].
        # The `y_points` array (convolution + 2 zeros) has `num_conv_points + 2` points,
        # defining `num_conv_points + 1` intervals over the domain [0,2].
        h_conv = 2.0 / (num_conv_points + 1) # Step size for convolution on [0,2]

        # Calculate L2-norm squared of the convolution (rigorous method for piecewise linear)
        y_points = jnp.concatenate([jnp.array([0.0]), convolution, jnp.array([0.0])])
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((h_conv / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate integral_f = ∫f dx (using trapezoidal rule for consistency)
        # f_non_negative represents N samples of f on [0,1].
        # We assume f(0)=0 and f(1)=0 for integration, creating N+2 points defining N+1 intervals.
        f_points_for_integral = jnp.concatenate([jnp.array([0.0]), f_non_negative, jnp.array([0.0])])
        h_f = 1.0 / (N + 1) # Step size for f on [0,1] with N+1 intervals
        integral_f = jnp.sum((h_f / 2) * (f_points_for_integral[:-1] + f_points_for_integral[1:]))

        # Denominator term: (∫f)²
        integral_f_squared = integral_f**2
        
        # Ensure integral_f_squared is not too close to zero to avoid division by zero
        epsilon = 1e-10
        integral_f_squared = jnp.maximum(integral_f_squared, epsilon)

        # Calculate infinity-norm of the convolution
        # Since f(x) >= 0, convolution (f*f)(x) >= 0, so abs() is not strictly needed.
        norm_inf = jnp.max(convolution)
        # Ensure norm_inf is not too close to zero
        norm_inf = jnp.maximum(norm_inf, epsilon)
        
        # Calculate C2 ratio
        denominator = integral_f_squared * norm_inf
        c2_ratio = l2_norm_squared / denominator
        
        # Add a smoothness regularizer to discourage spiky/noisy solutions.
        # This penalizes the L2 norm of the discrete second derivative.
        second_derivative = f_non_negative[2:] - 2 * f_non_negative[1:-1] + f_non_negative[:-2]
        smoothness_penalty = jnp.sum(second_derivative**2)

        # We want to MAXIMIZE C2 (minimize -C2) and MINIMIZE the penalty.
        return -c2_ratio + self.hypers.regularization_strength * smoothness_penalty

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
        
        # Initialize f_values with a triangular pulse function
        # This provides a more structured and potentially better starting point
        # than random uniform initialization.
        # f(x) = 2x for x in [0, 0.5], f(x) = 2(1-x) for x in [0.5, 1]
        # These points correspond to x_i = (k+1)/(N+1) for k=0, ..., N-1
        x_coords = jnp.arange(1, self.hypers.num_intervals + 1) / (self.hypers.num_intervals + 1)
        f_values = jnp.where(x_coords <= 0.5, 2 * x_coords, 2 * (1 - x_coords))
        
        # Note: The 'key' for PRNGKey(42) is no longer directly used for f_values
        # but could be used for other stochastic components if introduced later.
        
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

