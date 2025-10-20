# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    # Multi-scale optimization parameters
    scales: tuple[int, ...] = (64, 128, 256, 512, 1024) # Sequence of resolutions to optimize on
    steps_per_scale: int = 25000 # Number of steps for each scale, total steps = sum(steps_per_scale)
    
    learning_rate: float = 0.005 # Adjusted learning rate, often better for higher resolution with more parameters
    warmup_steps: int = 1000 # Adjusted warmup steps proportionally for each scale
    
    regularization_strength: float = 1e-5 # Strength of L2 regularization on function differences

class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the objective function using the unitless norm calculation.
        f_values here are the raw values that will be squared to ensure non-negativity.
        Returns (loss, c2_ratio) where loss includes regularization.
        """
        f_non_negative = jnp.square(f_values) # Using square for non-negativity ensures smooth gradients and f >= 0
        
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
        denominator = norm_1 * norm_inf + 1e-12 # Add epsilon for numerical stability
        c2_ratio = l2_norm_squared / denominator
        
        # Add regularization for smoothness of f_non_negative
        # Penalize large differences between adjacent points
        regularization_term = jnp.sum(jnp.diff(f_non_negative)**2)
        
        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        # The loss includes the regularization term.
        loss = -c2_ratio + self.hypers.regularization_strength * regularization_term
        return loss, c2_ratio # Return both the loss and the actual C2 ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        # Use has_aux=True to get both the loss and the c2_ratio from _objective_fn
        (loss_val, c2_ratio), grads = jax.value_and_grad(self._objective_fn, has_aux=True)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss_val, c2_ratio # Return c2_ratio for logging

    def run_optimization(self, num_intervals: int, initial_f_values: jnp.ndarray = None, random_seed: int = 42):
        """
        Sets up and runs the full optimization process for a given number of intervals.
        Can be initialized with previous f_values for multi-scale optimization.
        `initial_f_values` should be the *squared* (non-negative) function values from the previous scale.
        """
        self.hypers.num_intervals = num_intervals # Update num_intervals for current scale
        num_steps_current_scale = self.hypers.steps_per_scale
        
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=num_steps_current_scale - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-4
        )
        self.optimizer = optax.adam(learning_rate=schedule)
        
        key = jax.random.PRNGKey(random_seed)
        if initial_f_values is None:
            # Random initialization for the first scale (raw values before squaring)
            f_values = jax.random.uniform(key, (num_intervals,))
        else:
            # Interpolate f_values from previous scale for refinement
            # initial_f_values are the non-negative f(x) values (i.e., square(raw_f_values))
            x_old = jnp.linspace(0, 1, initial_f_values.shape[0], endpoint=False)
            x_new = jnp.linspace(0, 1, num_intervals, endpoint=False)
            
            # Simple linear interpolation of the non-negative function values
            interpolated_f_non_negative = jnp.interp(x_new, x_old, initial_f_values)
            
            # Convert back to the "raw" values for optimization (take square root)
            f_values = jnp.sqrt(interpolated_f_non_negative) 
            
            # Ensure strictly positive to avoid sqrt(0) issues and dead gradients if values are exactly zero
            f_values = jnp.clip(f_values, 1e-6, None) 
            
            # Add some noise to escape local minima, especially important when refining
            f_values += jax.random.normal(key, f_values.shape) * 1e-3
            f_values = jnp.clip(f_values, 1e-6, None) # Clip again after noise

        # Initialize opt_state with the current f_values
        opt_state = self.optimizer.init(f_values)
        print(f"\n--- Optimizing for N={num_intervals} (Steps: {num_steps_current_scale}) ---")
        train_step_jit = jax.jit(self.train_step)

        loss_val = jnp.inf
        c2_ratio_step = 0.0
        for step in range(num_steps_current_scale):
            f_values, opt_state, loss_val, c2_ratio_step = train_step_jit(f_values, opt_state)
            if step % (num_steps_current_scale // 10) == 0 or step == num_steps_current_scale - 1:
                # Print the actual C2 ratio, not the loss including regularization
                print(f"  Step {step:5d} | C2 ≈ {c2_ratio_step:.8f}")
        
        # Calculate final C2 value (without regularization) for the optimized f_values
        final_c2 = self._objective_fn(f_values)[1]
        print(f"  Final C2 for N={num_intervals}: {final_c2:.8f}")
        return jnp.square(f_values), final_c2 # Return the actual f(x) values (squared raw_f_values)

def run():
    """Entry point for running the multi-scale optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    
    current_f_values = None # Stores the non-negative f(x) values (squared) from the previous scale
    final_c2_val = 0.0
    
    for i, n_intervals in enumerate(hypers.scales):
        # Pass the optimized f_values from the previous scale as initial_f_values
        # For the first scale, current_f_values will be None, triggering random init.
        # random_seed is varied per scale to introduce more exploration while maintaining overall reproducibility.
        current_f_values, final_c2_val_current_scale = optimizer.run_optimization(
            num_intervals=n_intervals,
            initial_f_values=current_f_values,
            random_seed=42 + i 
        )
        final_c2_val = final_c2_val_current_scale # Keep track of the C2 from the highest resolution

    # Recalculate the final loss value using the objective function with regularization
    # This is for consistency with the 'loss' metric, which includes regularization.
    # Note: the final_c2_val is the unregularized C2 from the last scale.
    # We need to set the num_intervals in hypers to the final scale for correct _objective_fn call.
    optimizer.hypers.num_intervals = hypers.scales[-1]
    final_raw_f_values = jnp.sqrt(current_f_values) # Convert back to raw for _objective_fn
    loss_val = optimizer._objective_fn(final_raw_f_values)[0] # Get the loss part (with regularization)
    
    f_values_np = np.array(current_f_values) # This will be the f_non_negative from the last scale
    
    return f_values_np, float(final_c2_val), float(loss_val), hypers.scales[-1] # Return num_intervals of the last scale
# EVOLVE-BLOCK-END

