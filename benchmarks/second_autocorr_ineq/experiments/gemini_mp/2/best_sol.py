# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    num_intervals: int = 1024  # Increased resolution for f(x) for higher fidelity. Power of 2 is FFT-friendly.
    learning_rate: float = 0.01
    num_steps: int = 120000  # More steps for convergence with more parameters and regularization.
    warmup_steps: int = 10000 # Proportional increase in warmup steps for longer training.
    regularization_strength: float = 1e-6 # L2 regularization on function differences (smoothness).
    # New hyperparameter: L2 regularization on boundary values of f to encourage compact support.
    regularization_strength_boundary: float = 1e-5 

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
        # This replaces the simple Riemann sum with a more accurate integration method.
        # ||f*f||_1 = (∫f)^2
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
        
        # Add L2 regularization on the differences of f_non_negative to promote smoothness
        regularization_term_smoothness = jnp.sum(jnp.diff(f_non_negative)**2)
        
        # Add L2 regularization on the boundary values of f_non_negative to encourage compact support.
        # This helps guide the function to be zero at the edges of its assumed domain [0,1].
        regularization_term_boundary = f_non_negative[0]**2 + f_non_negative[-1]**2
        
        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        # Add regularization terms to the loss (negative C2 ratio).
        return -c2_ratio + \
               self.hypers.regularization_strength * regularization_term_smoothness + \
               self.hypers.regularization_strength_boundary * regularization_term_boundary

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
            end_value=self.hypers.learning_rate * 1e-5 # Lower end_value for finer convergence
        )
        self.optimizer = optax.adam(learning_rate=schedule)
        
        key = jax.random.PRNGKey(42)
        # Reverting to random normal initialization as it showed better empirical performance
        # Initialize log-parameters from a normal distribution for exp parameterization.
        f_params = jax.random.normal(key, (self.hypers.num_intervals,))
        
        opt_state = self.optimizer.init(f_params)
        print(f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}")
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_params, opt_state, loss = train_step_jit(f_params, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                # The reported loss includes the regularization term, so for C2 we need to subtract it.
                # Re-evaluate the objective without regularization for accurate C2 reporting.
                # Create a temporary hypers object with zero regularization to get the 'pure' C2 value.
                # IMPORTANT: Create a new Hyperparameters instance to avoid modifying self.hypers
                temp_hypers_no_reg = Hyperparameters(
                    num_intervals=self.hypers.num_intervals,
                    learning_rate=self.hypers.learning_rate,
                    num_steps=self.hypers.num_steps,
                    warmup_steps=self.hypers.warmup_steps,
                    regularization_strength=0.0, # Set smoothness reg to 0
                    regularization_strength_boundary=0.0 # Set boundary reg to 0
                )
                current_c2_val = -C2Optimizer(temp_hypers_no_reg)._objective_fn(f_params)
                print(f"Step {step:5d} | C2 (approx) ≈ {current_c2_val:.8f}")
        
        # Final C2 calculation should also be without regularization
        # IMPORTANT: Create a new Hyperparameters instance to avoid modifying self.hypers
        temp_hypers_no_reg = Hyperparameters(
            num_intervals=self.hypers.num_intervals,
            learning_rate=self.hypers.learning_rate,
            num_steps=self.hypers.num_steps,
            warmup_steps=self.hypers.warmup_steps,
            regularization_strength=0.0, # Set smoothness reg to 0
            regularization_strength_boundary=0.0 # Set boundary reg to 0
        )
        final_c2 = -C2Optimizer(temp_hypers_no_reg)._objective_fn(f_params)
        
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        # Return the actual function values, not the log-parameters.
        return jnp.exp(f_params), final_c2

def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()
    
    # The loss value reported from the objective function includes regularization.
    # The true C2 value is calculated by re-evaluating the objective without regularization.
    # To get the 'loss' metric without regularization for comparison:
    # Create a temporary optimizer instance with regularization_strength set to 0 for this specific calculation.
    # IMPORTANT: Create a new Hyperparameters instance to avoid modifying the original 'hypers' object
    optimizer_no_reg = C2Optimizer(Hyperparameters(
        num_intervals=hypers.num_intervals,
        learning_rate=hypers.learning_rate,
        num_steps=hypers.num_steps,
        warmup_steps=hypers.warmup_steps,
        regularization_strength=0.0, # Set smoothness reg to 0
        regularization_strength_boundary=0.0 # Set boundary reg to 0
    ))
    loss_val_no_reg = optimizer_no_reg._objective_fn(jnp.log(optimized_f))
    
    f_values_np = np.array(optimized_f)
    
    return f_values_np, float(final_c2_val), float(loss_val_no_reg), hypers.num_intervals
# EVOLVE-BLOCK-END

