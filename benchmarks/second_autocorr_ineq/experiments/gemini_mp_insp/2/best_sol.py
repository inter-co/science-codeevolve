# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    num_intervals: int = 512 # Increased for finer discretization
    num_segments: int = 2   # Number of characteristic functions to sum
    domain_length: float = 1.0 # L for the support of f
    learning_rate: float = 0.01
    num_steps: int = 30000  # Increased steps for more complex parameterization
    warmup_steps: int = 3000
    # Add a random seed for reproducibility
    seed: int = 42

class C2Optimizer:
    """
    Optimizes a parametric function (sum of characteristic functions of disjoint intervals)
    to find a lower bound for the C2 constant. This version enforces constraints
    algebraically for a more efficient and stable search.
    """
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        # Set JAX random seed for reproducibility
        self.key = jax.random.PRNGKey(self.hypers.seed)

    def _build_f_from_params(self, raw_params: jnp.ndarray) -> jnp.ndarray:
        """
        Constructs a normalized, discretized function f(x) from structured parameters.
        f(x) is a sum of characteristic functions of DISJOINT intervals.
        The parameters define the lengths of intervals/gaps and relative amplitudes.
        The function is normalized such that ∫f(x)dx = 1 by construction.
        """
        k = self.hypers.num_segments
        # Unpack raw_params: (2k+1) for lengths, k for amplitudes
        raw_lengths = raw_params[:2*k + 1]
        raw_amplitudes = raw_params[2*k + 1:]

        # 1. Determine interval boundaries from lengths
        # Use softmax to ensure lengths sum to L and are positive. This creates a
        # partition of the domain [0, L] into k intervals and k+1 gaps.
        L = self.hypers.domain_length
        lengths = jax.nn.softmax(raw_lengths) * L
        
        # Boundaries are cumulative sums of lengths
        boundaries = jnp.cumsum(lengths)
        
        # Extract start/end points for the k intervals
        # s_i = boundaries[2*i], e_i = boundaries[2*i+1]
        starts = boundaries[0:-1:2]
        ends = boundaries[1::2]
        
        # Actual interval lengths for normalization
        interval_lengths = ends - starts

        # 2. Determine normalized amplitudes
        # Use softplus for non-negative relative amplitudes, providing better gradients than relu
        relative_amplitudes = jax.nn.softplus(raw_amplitudes)
        
        # Normalization factor: sum(relative_amp_i * length_i)
        norm_factor = jnp.sum(relative_amplitudes * interval_lengths)
        
        # Final amplitudes A_i such that sum(A_i * length_i) = 1
        final_amplitudes = relative_amplitudes / (norm_factor + 1e-9)

        # 3. Construct f(x) on the grid (vectorized)
        N = self.hypers.num_intervals
        x_grid = jnp.linspace(0, L, N, endpoint=False)
        
        # Reshape for broadcasting: x_grid (N,) -> (N,1); starts/ends/amps (k,)
        indicators = (x_grid[:, None] >= starts) & (x_grid[:, None] < ends)
        f_values = jnp.sum(indicators * final_amplitudes, axis=1)
            
        return f_values

    def _objective_fn(self, raw_params: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function C₂ = ||f ★ f||₂² / ||f ★ f||_{∞}.
        The function f is constructed from parameters to be pre-normalized (∫f=1).
        """
        # 1. Build the pre-normalized f_values from raw_params
        f_normalized = self._build_f_from_params(raw_params)
        
        # 2. Compute g = f ★ f using FFT
        N = self.hypers.num_intervals
        L = self.hypers.domain_length
        fft_size = 2 * N
        padded_f = jnp.pad(f_normalized, (0, N))
        fft_f = jnp.fft.fft(padded_f, n=fft_size)
        g = jnp.fft.ifft(fft_f * fft_f, n=fft_size).real

        # 3. Compute norms of g
        h_g = (2 * L) / fft_size
        l2_norm_squared = jnp.sum(g**2) * h_g 
        norm_inf = jnp.max(g)

        # 4. Compute C2
        c2 = l2_norm_squared / (norm_inf + 1e-9)
        
        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2

    def train_step(self, raw_params: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(raw_params)
        updates, opt_state = self.optimizer.update(grads, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)
        return raw_params, opt_state, loss

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
        
        # Initialize raw_params.
        # For k segments, we have 2k+1 lengths (intervals + gaps) and k amplitudes.
        k = self.hypers.num_segments
        num_params = (2 * k + 1) + k
        self.key, subkey = jax.random.split(self.key)
        # Initialize with small random normal values. This encourages softmax/softplus
        # to start with roughly uniform lengths and amplitudes.
        raw_params = jax.random.normal(subkey, (num_params,)) * 0.01

        opt_state = self.optimizer.init(raw_params)
        print(f"Number of intervals (N): {self.hypers.num_intervals}, Segments: {self.hypers.num_segments}, Steps: {self.hypers.num_steps}")
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            raw_params, opt_state, loss = train_step_jit(raw_params, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.8f}")
        
        final_c2 = -self._objective_fn(raw_params)
        
        # Reconstruct the final f_values for output. It's already normalized by construction.
        optimized_f = self._build_f_from_params(raw_params)

        print(f"Final C2 lower bound found: {final_c2:.8f}")
        return optimized_f, final_c2

def run():
    """Entry point for running the optimization."""
    # Ensure numpy seed is also set for any numpy operations outside JAX
    np.random.seed(42) 
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()
    
    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)
    
    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals
# EVOLVE-BLOCK-END

