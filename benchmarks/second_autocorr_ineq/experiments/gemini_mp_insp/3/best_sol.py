# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

# Enable 64-bit precision for higher numerical stability, crucial for sensitive optimization.
jax.config.update("jax_enable_x64", True)

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    num_intervals: int = 2048 # Increased resolution (from Inspiration 1)
    learning_rate: float = 0.005
    num_steps: int = 200000 # Increased for more thorough optimization (from Inspiration 1)
    warmup_steps: int = 20000 # Adjusted proportionally to num_steps (from Inspiration 1)
    domain_length_f: float = 2.0 # Crucial change: f(x) is supported on [0, L]. L=2.0 performed best in IP1.
    regularization_lambda: float = 5e-6 # Slightly reduced L2 regularization for potentially sharper solutions
    init_type: str = "triangular" # "uniform", "triangular", "raised_cosine", "gaussian_bimodal"

class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant.
    This implementation rigorously follows the mathematical framework by
    normalizing f such that its integral is 1, simplifying the objective.
    (Incorporates best practices from Inspiration 1 & 2)
    """
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.N = self.hypers.num_intervals
        self.L = self.hypers.domain_length_f
        self.dx = self.L / self.N # Grid spacing for f(x) on its domain [0, L]

    def _calculate_c2_metric(self, f_raw_params: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the pure C2 constant from raw function parameters, without regularization.
        Uses jax.lax.cond for JIT-friendly conditional logic. (Adapted from Inspiration 2)
        """
        f_non_negative = jax.nn.relu(f_raw_params)
        
        # Trapezoidal rule for integral_f
        integral_f = jnp.sum(f_non_negative[1:] + f_non_negative[:-1]) * self.dx / 2.0
        
        is_trivial_f = integral_f < 1e-9

        # Define the C2 calculation for a non-trivial function
        def compute_c2_for_non_trivial():
            # Normalize f(x) such that integral f(x) dx = 1. Add epsilon for robustness.
            f_normalized = f_non_negative / (integral_f + 1e-9)
            
            # Compute discrete convolution (f * f)(x) via FFT
            padded_f = jnp.pad(f_normalized, (0, self.N)) # Pad with N zeros to make length 2N
            fft_f = jnp.fft.fft(padded_f)
            raw_discrete_conv = jnp.fft.ifft(fft_f * fft_f).real
            convolution_approx = self.dx * raw_discrete_conv # Scale to approximate continuous integral

            # Calculate ||f * f||_2^2 (Trapezoidal rule)
            l2_norm_sq_conv = jnp.sum(convolution_approx[1:]**2 + convolution_approx[:-1]**2) * self.dx / 2.0

            # Calculate ||f * f||_inf (peak value). Add epsilon for robustness.
            norm_inf_conv = jnp.max(convolution_approx)
            norm_inf_conv_safe = norm_inf_conv + 1e-9
            
            c2_constant = l2_norm_sq_conv / norm_inf_conv_safe
            return c2_constant

        # Use jax.lax.cond for clean, JIT-compatible conditional logic (from Inspiration 2)
        return jax.lax.cond(is_trivial_f, lambda: jnp.array(0.0), compute_c2_for_non_trivial)

    def _objective_fn(self, f_raw_params: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function (negative C2 + regularization) based on raw parameters.
        Uses Trapezoidal rule for higher accuracy integration. (Adapted from Inspiration 2)
        """
        f_non_negative = jax.nn.relu(f_raw_params)
        integral_f = jnp.sum(f_non_negative[1:] + f_non_negative[:-1]) * self.dx / 2.0
        is_trivial_f = integral_f < 1e-9
        
        def calculate_loss_for_non_trivial():
            # Normalize f(x) such that integral f(x) dx = 1. Add epsilon for robustness.
            f_normalized = f_non_negative / (integral_f + 1e-9)
            
            # Compute C2 constant
            c2_constant = self._calculate_c2_metric(f_raw_params) # Re-use the C2 metric calculation
            
            # Regularization term: L2 norm of differences in the normalized function for smoothness
            smoothness_reg = jnp.sum(jnp.diff(f_normalized)**2)
            
            # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
            return -c2_constant + self.hypers.regularization_lambda * smoothness_reg

        # If f_values were trivial, return a very high loss to guide optimization away.
        return jax.lax.cond(is_trivial_f, lambda: jnp.array(1e9), calculate_loss_for_non_trivial)

    def train_step(self, f_raw_params: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step, optimizing raw function parameters."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_raw_params)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_raw_params)
        f_raw_params = optax.apply_updates(f_raw_params, updates)
        return f_raw_params, opt_state, loss

    def run_optimization(self, key: jax.random.PRNGKey):
        """Sets up and runs the full optimization process."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-4
        )
        self.optimizer = optax.adam(learning_rate=schedule)
        
        # Initialize f_raw_params based on init_type
        if self.hypers.init_type == "uniform":
            f_raw_params = jax.random.uniform(key, (self.N,))
        elif self.hypers.init_type == "triangular":
            key, subkey = jax.random.split(key)
            x_vals = jnp.linspace(0, self.L, self.N, endpoint=False)
            mid_point = self.L / 2.0
            slope = 4.0 / (self.L * self.L) # For a triangle with base L to integrate to 1
            initial_f_values = jnp.where(x_vals < mid_point, 
                                 slope * x_vals, 
                                 slope * (self.L - x_vals))
            f_raw_params = initial_f_values + 0.01 * jax.random.uniform(subkey, (self.N,)) # Add small perturbation
        elif self.hypers.init_type == "raised_cosine":
            key, subkey = jax.random.split(key)
            x_vals = jnp.linspace(0, self.L, self.N, endpoint=False)
            initial_f_values = (1 + jnp.cos(jnp.pi * (x_vals - self.L/2) / (self.L/2))) / self.L
            f_raw_params = initial_f_values + 0.05 * jax.random.uniform(subkey, (self.N,)) # Increased perturbation for raised_cosine
        elif self.hypers.init_type == "gaussian_bimodal":
            key, subkey = jax.random.split(key)
            x_vals = jnp.linspace(0, self.L, self.N, endpoint=False)
            mean1, std_dev1 = self.L * 0.25, self.L * 0.1
            pulse1 = jnp.exp(-0.5 * ((x_vals - mean1) / std_dev1)**2)
            mean2, std_dev2 = self.L * 0.75, self.L * 0.1
            pulse2 = jnp.exp(-0.5 * ((x_vals - mean2) / std_dev2)**2)
            initial_f_values = pulse1 + pulse2
            f_raw_params = initial_f_values + 0.05 * jax.random.uniform(subkey, (self.N,)) # Increased perturbation for gaussian_bimodal
        else:
            raise ValueError(f"Unknown initialization type: {self.hypers.init_type}")
        
        opt_state = self.optimizer.init(f_raw_params)
        print(f"N: {self.N}, Steps: {self.hypers.num_steps}, L: {self.L}, Init: {self.hypers.init_type}, Reg: {self.hypers.regularization_lambda}")
        train_step_jit = jax.jit(self.train_step)
        get_c2_metric_jit = jax.jit(self._calculate_c2_metric) # JIT compile C2 metric calculation (from Inspiration 2)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_raw_params, opt_state, loss = train_step_jit(f_raw_params, opt_state)
            
            if step % 10000 == 0 or step == self.hypers.num_steps - 1: # Print less frequently (from Inspiration 1)
                current_c2 = get_c2_metric_jit(f_raw_params) # Use JIT-compiled metric for logging
                print(f"Step {step:6d} | C2 ≈ {current_c2:.8f} | Loss = {loss:.8f}")

        # Final evaluation: calculate pure C2 and the final normalized function (from Inspiration 1)
        final_f_non_negative = jax.nn.relu(f_raw_params)
        integral_f_final = jnp.sum(final_f_non_negative[1:] + final_f_non_negative[:-1]) * self.dx / 2.0
        final_c2 = 0.0
        final_f_normalized = jnp.zeros_like(final_f_non_negative) # Initialize to zeros

        # Recalculate C2 using the normalized function, ensuring it's the pure C2 (without regularization)
        # The _calculate_c2_metric already handles trivial functions (returns 0.0).
        final_c2 = get_c2_metric_jit(f_raw_params) 
        
        if integral_f_final > 1e-9:
            final_f_normalized = final_f_non_negative / (integral_f_final + 1e-9)
            
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        
        return final_f_normalized, final_c2 # Return the L1-normalized function (from Inspiration 1)

def run():
    """Entry point for running the optimization."""
    # Ensure reproducibility for all stochastic components
    np.random.seed(42)
    key = jax.random.PRNGKey(42)

    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization(key)
    
    # The reported loss should be the negative of the *final_c2_val* (without regularization)
    loss_val = -final_c2_val 
    f_values_np = np.array(optimized_f)
    
    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals
# EVOLVE-BLOCK-END

