# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    num_intervals: int = 256 # Increased resolution for better accuracy, matching AlphaEvolve scale
    learning_rate: float = 0.005 # Adjusted learning rate, often better for higher resolution with more parameters
    num_steps: int = 100000 # Increased steps for convergence with more parameters
    warmup_steps: int = 5000 # Adjusted warmup steps proportionally

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

        # Calculate L2-norm squared of the convolution (rigorous method for piecewise linear)
        # Assuming f is on [0, 1] with N intervals, its convolution f*f is on [0, 2].
        # The 'convolution' array has 2*N points, representing f*f at x_k = k * (1/N) for k=0 to 2N-1.
        # So the integration step size for f*f is dx_conv = 1.0 / N.
        dx_conv = 1.0 / N
        
        # To integrate over the full domain [0, 2], we need 2N+1 points (2N intervals).
        # The 'convolution' array has 2N points (g(0) to g(2-1/N)). We append g(2)=0,
        # which is the expected value for compactly supported functions at the domain end.
        y_conv_extended = jnp.concatenate([convolution, jnp.array([0.0])])
        
        # Use piecewise linear integration formula for g(x)^2: Integral(g^2) = (dx/3) * sum(y_k^2 + y_k y_{k+1} + y_{k+1}^2)
        y1_sq = y_conv_extended[:-1]**2
        y2_sq = y_conv_extended[1:]**2
        y1y2 = y_conv_extended[:-1] * y_conv_extended[1:]
        
        l2_norm_squared = jnp.sum( (dx_conv / 3) * (y1_sq + y1y2 + y2_sq) )

        # Calculate L1-norm using the simplification: ||f ★ f||₁ = (∫f)²
        # Assuming f_values represent f on [0, 1] with N intervals.
        # ∫f dx = sum(f_values) * dx_f = sum(f_values) * (1/N)
        dx_f = 1.0 / N
        integral_f = jnp.sum(f_non_negative) * dx_f
        norm_1 = integral_f**2

        # Calculate infinity-norm of the convolution. Since f >= 0, f*f >= 0, so abs is not needed.
        norm_inf = jnp.max(convolution)
        
        # Calculate C2 ratio
        denominator = norm_1 * norm_inf
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
        f_values = jax.random.uniform(key, (self.hypers.num_intervals,))
        
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

