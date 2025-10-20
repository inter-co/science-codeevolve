# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    # Increased resolution to leverage 64-bit precision and capture finer details.
    # 2049 = 2^11 + 1, often a good choice for symmetric functions with FFT.
    num_intervals: int = 2049
    learning_rate: float = 0.01 # Adjusted learning rate, as in Inspiration 1.
    # Increased steps for a longer, more thorough optimization run.
    num_steps: int = 300000
    # Adjusted proportionally to num_steps.
    warmup_steps: int = 30000
    # Add domain_length_f: L, the length of the domain for f(x)
    domain_length_f: float = 1.0 # f(x) is supported on [0, L]
    # Coefficient for smoothness regularization. A slightly higher value can encourage
    # smoother functions, potentially leading to better C2 values by avoiding
    # sharp, numerically unstable features, while still allowing flexibility.
    # Updated to match Inspiration Programs 1 and 2 for optimal performance.
    smoothness_coeff: float = 5e-9
    # Coefficient for symmetry regularization, encouraging f(x) = f(L-x).
    symmetry_coeff: float = 1e-7
    # Coefficient for boundary regularization, encouraging f(0) = f(L) = 0.
    boundary_coeff: float = 1e-6

class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.L = hypers.domain_length_f  # Domain length for f(x) (f has compact support on [0, L])
        # Grid spacing for f(x) on its domain [0, L]. Use L / (N - 1) for N points and trapezoidal rule.
        self.dx_f = self.L / (self.hypers.num_intervals - 1)

    def _calculate_c2_and_f_norm(self, f_values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Helper to compute C2 and the normalized function, for use in objective and final reporting."""
        N = self.hypers.num_intervals
        L = self.hypers.domain_length_f
        # dx_f is already L / (N - 1) from __init__
        dx_f = self.dx_f

        f_non_negative = jax.nn.relu(f_values)
        # Trapezoidal rule for integral of f: ∫f(x) dx ≈ dx * (Σ f_i - 0.5 * (f_0 + f_N-1))
        integral_f = dx_f * (jnp.sum(f_non_negative) - 0.5 * (f_non_negative[0] + f_non_negative[-1]))
        f_normalized = f_non_negative / (integral_f + 1e-9)

        # Convolution setup
        # The linear convolution of two N-point functions has length 2N-1.
        conv_len = 2 * N - 1 
        # Pad f_normalized (length N) to 2N for FFT-based circular convolution.
        # This allows accurate linear convolution by taking the first 2N-1 points.
        padded_f = jnp.pad(f_normalized, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        # Take real part and slice to actual linear convolution length (2N-1).
        raw_discrete_conv = jnp.fft.ifft(fft_f * fft_f).real[:conv_len]
        # Scale by dx_f to approximate the continuous convolution integral.
        convolution_approx = dx_f * raw_discrete_conv

        # The convolution g = f ★ f has support [0, 2L].
        # With 2N-1 points, the step size for g is dx_g = (2L) / ((2N-1) - 1) = (2L) / (2N-2) = L / (N-1).
        # This means dx_g is equal to dx_f.
        dx_conv = dx_f
        conv_squared = convolution_approx**2
        # Trapezoidal rule for L2 norm squared of convolution: ∫g(x)² dx ≈ dx_g * (Σ g_i² - 0.5 * (g_0² + g_last²))
        l2_norm_sq_conv = dx_conv * (jnp.sum(conv_squared) - 0.5 * (conv_squared[0] + conv_squared[-1]))

        norm_inf_conv = jnp.max(convolution_approx)
        norm_inf_conv_safe = norm_inf_conv + 1e-9

        c2_constant = l2_norm_sq_conv / norm_inf_conv_safe
        return c2_constant, f_normalized

    def _objective_fn(self, f_raw_params: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function to maximize C2, with added regularization penalties.
        """
        # Ensure f_raw_params are non-negative for C2 calculation within this function.
        # The `train_step` will also enforce non-negativity after updates.
        c2_constant, f_normalized = self._calculate_c2_and_f_norm(jax.nn.relu(f_raw_params))
        
        # 1. Smoothness penalty: Encourages smoother functions.
        smoothness_penalty = self.hypers.smoothness_coeff * jnp.sum(jnp.diff(f_normalized)**2)
        
        # 2. Symmetry penalty: Enforces f(x) ≈ f(L-x) by penalizing the difference
        # between the function and its flipped version, a known property of optimal solutions.
        symmetry_penalty = self.hypers.symmetry_coeff * jnp.sum((f_normalized - jnp.flip(f_normalized))**2)

        # 3. Boundary penalty: Encourages the function to be zero at the boundaries of its
        # compact support, enforcing f(0) = f(L) = 0 more strictly.
        boundary_penalty = self.hypers.boundary_coeff * (f_normalized[0]**2 + f_normalized[-1]**2)
        
        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        # The total loss is the negative C2 plus all regularization penalties.
        total_loss = -c2_constant + smoothness_penalty + symmetry_penalty + boundary_penalty
        return total_loss

    def train_step(self, f_raw_params: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step, optimizing raw function parameters."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_raw_params)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_raw_params)
        f_raw_params = optax.apply_updates(f_raw_params, updates)
        # Explicitly project f_raw_params to be non-negative after each update.
        # This ensures the constraint f(x) >= 0 is strictly maintained and prevents
        # the optimizer from hiding negative values, leading to more consistent results.
        f_raw_params = jax.nn.relu(f_raw_params)
        return f_raw_params, opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full optimization process."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-4
        )
        self.optimizer = optax.adamw(learning_rate=schedule) # Use AdamW for regularization
        
        N = self.hypers.num_intervals
        L = self.hypers.domain_length_f
        x = jnp.linspace(0, L, N)
        # Initialize f_values with a triangular function, often a good starting point for optimal shapes.
        # This initial shape is then perturbed with small uniform noise.
        f_raw_params = jnp.where(x <= L / 2, 2 * x / L, 2 * (L - x) / L)
        
        key = jax.random.PRNGKey(42)
        # Add small uniform noise to break perfect symmetry and explore space better.
        f_raw_params += 0.001 * jax.random.uniform(key, f_raw_params.shape, dtype=f_raw_params.dtype)

        opt_state = self.optimizer.init(f_raw_params)
        print(f"Number of intervals (N): {N}, Steps: {self.hypers.num_steps}, Domain Length (L): {L}")
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        log_interval = self.hypers.num_steps // 20 # Log progress 20 times during training
        if log_interval == 0: log_interval = 1

        for step in range(self.hypers.num_steps):
            f_raw_params, opt_state, loss = train_step_jit(f_raw_params, opt_state)
            if step % log_interval == 0 or step == self.hypers.num_steps - 1:
                # For logging, we evaluate the pure C2, without the penalty, for a clear view of progress.
                pure_c2, _ = self._calculate_c2_and_f_norm(f_raw_params)
                print(f"Step {step:7d} | C2 (pure) ≈ {pure_c2:.15f} | Loss (with penalty) = {loss:.15f}")
        
        # Final evaluation of C2, calculated without the penalty term for a pure result.
        final_c2, final_f_normalized = self._calculate_c2_and_f_norm(f_raw_params)
        print(f"Final C2 lower bound found: {final_c2:.15f}")
        # final_f_normalized is already non-negative and L1-normalized from _calculate_c2_and_f_norm
        return final_f_normalized, final_c2

def run():
    """Entry point for running the optimization."""
    # Enable 64-bit precision for higher numerical accuracy during optimization.
    jax.config.update("jax_enable_x64", True)
    
    # Ensure reproducibility for all stochastic components
    np.random.seed(42)

    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()
    
    # The final loss from the training loop includes the penalty, so it's not a pure -C2.
    # For consistency with the provided metrics reporting, `loss_val` is set to `-final_c2_val`.
    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)
    
    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals
# EVOLVE-BLOCK-END

