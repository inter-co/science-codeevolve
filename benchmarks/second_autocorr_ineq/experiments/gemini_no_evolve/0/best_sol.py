# EVOLVE-BLOCK-START
import jax
# Enable 64-bit precision for JAX computations for higher accuracy
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    # Significantly increased resolution and steps for finding finer-grained solutions.
    # N=1024 is a common resolution for such problems, balancing accuracy and computation.
    num_intervals: int = 1024 # Increased from 250
    learning_rate: float = 0.005
    num_steps: int = 100000 # Increased from 40000
    warmup_steps: int = 10000 # Increased from 4000, roughly proportional to num_steps

class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _get_f_values(self, params: jnp.ndarray) -> jnp.ndarray:
        """
        Transforms unconstrained parameters into non-negative f_values.
        Using params**2 ensures non-negativity and allows for smooth gradients
        even when params are near zero. A small epsilon is added for numerical stability
        to prevent exact zeros if necessary, although params**2 is differentiable at 0.
        """
        return params**2 + 1e-9 # Add a tiny epsilon to prevent f_values from being exactly zero

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using a corrected, physically-scaled norm calculation.
        This treats `f` as a piecewise constant function on [0, 1].
        """
        # f_values are already non-negative due to transformation in train_step
        N = self.hypers.num_intervals
        dx = 1.0 / N

        # ||f ★ f||₁ = (∫f)², where ∫f is approximated by a Riemann sum.
        integral_f = jnp.sum(f_values) * dx
        norm_1_conv = integral_f**2

        # Autoconvolution g = f ★ f is computed via FFT.
        # The result must be scaled by dx to approximate the continuous convolution integral.
        padded_f = jnp.pad(f_values, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution_unscaled = jnp.fft.ifft(fft_f * fft_f).real
        g = convolution_unscaled * dx

        # ||f ★ f||_{∞} is the max value of the resulting convolution.
        # g is non-negative since f is non-negative.
        norm_inf = jnp.max(g)

        # ||f ★ f||₂² = ∫(f ★ f)² is calculated using a high-accuracy
        # piecewise-linear integral formula. The step size 'h' is dx.
        h = dx
        # For piecewise linear integration of g^2 from x=0 to x=2,
        # we need g(0) to g(2N*dx), where g(2N*dx) is assumed to be 0.
        # The `g` array has 2N points, representing g(k*dx) for k=0 to 2N-1.
        # So we append a 0 for g(2N*dx) to correctly define the last interval.
        y_points = jnp.concatenate([g, jnp.array([0.0], dtype=g.dtype)])
        y1, y2 = y_points[:-1], y_points[1:] # y1: g[0]..g[2N-1], y2: g[1]..g[2N] (which is 0)
        l2_norm_squared = jnp.sum((h / 3) * (y1**2 + y1 * y2 + y2**2))
        
        # Calculate C2 = ||f ★ f||₂² / (||f ★ f||₁ * ||f ★ f||_{∞})
        # Add a small epsilon to the denominator to prevent division by zero.
        denominator = norm_1_conv * norm_inf + 1e-12
        c2 = l2_norm_squared / denominator
        
        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2

    def train_step(self, params: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step, optimizing unconstrained parameters."""
        # Define a loss function that transforms params to f_values before computing the objective.
        def loss_fn(current_params):
            f_values = self._get_f_values(current_params)
            return self._objective_fn(f_values)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        # No need for relu here, as f_values are inherently non-negative via _get_f_values
        return params, opt_state, loss

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
        N = self.hypers.num_intervals
        
        # --- Improved Initialization: Hat function for f_values_init ---
        # Initialize with a "hat" shape, which often performs better for such problems.
        # This provides a better starting point resembling optimal step functions.
        start_ramp_idx = N // 4
        end_ramp_idx = 3 * N // 4
        
        # Linear ramp up from 0 to 1
        ramp_up = jnp.linspace(0.0, 1.0, start_ramp_idx, dtype=jnp.float64)
        # Flat top at 1
        flat_top = jnp.ones(end_ramp_idx - start_ramp_idx, dtype=jnp.float64)
        # Linear ramp down from 1 to 0
        ramp_down = jnp.linspace(1.0, 0.0, N - end_ramp_idx, dtype=jnp.float64)
        
        f_values_init = jnp.concatenate([ramp_up, flat_top, ramp_down])

        # Initialize params such that params**2 ≈ f_values_init.
        # Add small epsilon to f_values_init before sqrt to avoid sqrt(0) issues,
        # ensuring params_init are non-zero where f_values_init is zero.
        params_init = jnp.sqrt(f_values_init + 1e-9)
        
        # Add a small amount of noise to break perfect symmetry and explore.
        # Ensure noise is also float64. Reduced noise factor slightly for stability.
        params = params_init + 0.005 * jax.random.uniform(key, params_init.shape, dtype=jnp.float64)
        
        opt_state = self.optimizer.init(params) # Initialize optimizer with params
        print(f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}")
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            params, opt_state, loss = train_step_jit(params, opt_state) # Pass params to train_step
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.8f}")
        
        # Calculate final C2 using the optimized parameters transformed to f_values
        final_f_values = self._get_f_values(params)
        final_c2 = -self._objective_fn(final_f_values)
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        return final_f_values, final_c2

def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()
    
    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)
    
    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals
# EVOLVE-BLOCK-END

