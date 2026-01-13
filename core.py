"""
Lenia-inspired morphogen field with external stimulus injection.
JAX-accelerated for real-time performance.
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class FieldConfig:
    """Configuration for the morphogen field."""
    # Grid size
    width: int = 512
    height: int = 512
    
    # Lenia kernel parameters
    kernel_radius: int = 13
    kernel_sigma: float = 0.5  # Controls kernel shape (beta in Lenia)
    
    # Growth function parameters (bell curve)
    growth_mu: float = 0.15    # Center of growth function
    growth_sigma: float = 0.015  # Width of growth function
    growth_amplitude: float = 1.0
    
    # Dynamics
    dt: float = 0.1           # Time step
    diffusion: float = 0.01   # Additional diffusion (beyond Lenia kernel)
    decay: float = 0.001      # Passive decay rate
    
    # Morphogen injection
    injection_radius: float = 5.0   # Radius of injection spot
    injection_power: float = 0.1    # Intensity per injection point
    
    # Performance
    use_fft: bool = True      # Use FFT convolution (faster for large kernels)


def make_kernel(config: FieldConfig) -> jnp.ndarray:
    """Create Lenia-style kernel (ring-shaped, normalized)."""
    r = config.kernel_radius
    size = 2 * r + 1
    
    # Create distance matrix
    y, x = jnp.mgrid[-r:r+1, -r:r+1]
    dist = jnp.sqrt(x**2 + y**2) / r  # Normalized to [0, 1]
    
    # Ring kernel with exponential falloff (Lenia style)
    kernel = jnp.exp(-((dist - 0.5) ** 2) / (2 * config.kernel_sigma ** 2))
    kernel = jnp.where(dist <= 1.0, kernel, 0.0)
    
    # Normalize
    kernel = kernel / jnp.sum(kernel)
    
    return kernel


def make_diffusion_kernel(size: int = 3) -> jnp.ndarray:
    """Simple Laplacian kernel for diffusion."""
    if size == 3:
        kernel = jnp.array([
            [0.05, 0.2, 0.05],
            [0.2, -1.0, 0.2],
            [0.05, 0.2, 0.05]
        ])
    else:
        # Larger Gaussian-weighted Laplacian
        r = size // 2
        y, x = jnp.mgrid[-r:r+1, -r:r+1]
        dist = jnp.sqrt(x**2 + y**2)
        kernel = jnp.exp(-dist**2 / (2 * (r/2)**2))
        kernel = kernel / jnp.sum(kernel)
        center = jnp.zeros_like(kernel).at[r, r].set(1.0)
        kernel = kernel - center
    
    return kernel


def growth_function(u: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
    """Lenia growth function (bell curve centered at mu)."""
    return 2.0 * jnp.exp(-((u - mu) ** 2) / (2 * sigma ** 2)) - 1.0


@partial(jit, static_argnums=(2, 3))
def convolve_fft(field: jnp.ndarray, kernel: jnp.ndarray, 
                  field_shape: Tuple[int, int], kernel_shape: Tuple[int, int]) -> jnp.ndarray:
    """FFT-based convolution with periodic boundaries."""
    # Pad kernel to field size
    pad_h = field_shape[0] - kernel_shape[0]
    pad_w = field_shape[1] - kernel_shape[1]
    
    kernel_padded = jnp.pad(kernel, 
                           ((0, pad_h), (0, pad_w)), 
                           mode='constant')
    
    # Roll to center the kernel
    kernel_padded = jnp.roll(kernel_padded, 
                            (-kernel_shape[0]//2, -kernel_shape[1]//2), 
                            axis=(0, 1))
    
    # FFT convolution
    field_fft = jnp.fft.fft2(field)
    kernel_fft = jnp.fft.fft2(kernel_padded)
    result = jnp.fft.ifft2(field_fft * kernel_fft)
    
    return jnp.real(result)


@partial(jit, static_argnums=(3,))
def create_injection_mask(positions: jnp.ndarray, 
                          powers: jnp.ndarray,
                          shape: Tuple[int, int],
                          radius: float) -> jnp.ndarray:
    """Create injection mask from positions.
    
    Args:
        positions: (N, 2) array of (x, y) positions
        powers: (N,) array of injection powers per point
        shape: (height, width) of field
        radius: injection spot radius
    """
    h, w = shape
    y_grid, x_grid = jnp.mgrid[0:h, 0:w]
    
    mask = jnp.zeros((h, w))
    
    def add_point(mask, pos_power):
        pos, power = pos_power[:2], pos_power[2]
        x, y = pos[0], pos[1]
        dist = jnp.sqrt((x_grid - x)**2 + (y_grid - y)**2)
        contribution = power * jnp.exp(-(dist**2) / (2 * radius**2))
        return mask + contribution
    
    # Vectorized version using scan
    if positions.shape[0] > 0:
        pos_powers = jnp.concatenate([positions, powers[:, None]], axis=1)
        mask, _ = jax.lax.scan(lambda m, pp: (add_point(m, pp), None), 
                               mask, pos_powers)
    
    return mask


class LeniaField:
    """Main class for running the morphogen field simulation."""
    
    def __init__(self, config: Optional[FieldConfig] = None):
        self.config = config or FieldConfig()
        self.reset()
        
        # Pre-compile kernels
        self.lenia_kernel = make_kernel(self.config)
        self.diffusion_kernel = make_diffusion_kernel()
        
        # JIT compile the step function
        self._step_jit = jit(self._step_impl)
        
    def reset(self):
        """Reset field to zero."""
        self.field = jnp.zeros((self.config.height, self.config.width))
        
    def set_field(self, field: np.ndarray):
        """Set field from numpy array."""
        self.field = jnp.array(field)
        
    def _step_impl(self, field: jnp.ndarray, 
                   injection_mask: jnp.ndarray,
                   lenia_kernel: jnp.ndarray,
                   dt: float,
                   diffusion: float,
                   decay: float,
                   growth_mu: float,
                   growth_sigma: float,
                   growth_amplitude: float) -> jnp.ndarray:
        """Single step of the simulation."""
        
        h, w = field.shape
        kh, kw = lenia_kernel.shape
        
        # Lenia-style update: convolve then apply growth function
        if self.config.use_fft:
            potential = convolve_fft(field, lenia_kernel, (h, w), (kh, kw))
        else:
            # Direct convolution (slower but sometimes more stable)
            potential = jax.scipy.signal.convolve2d(
                field, lenia_kernel, mode='same', boundary='wrap'
            )
        
        # Growth function
        growth = growth_function(potential, growth_mu, growth_sigma)
        
        # Additional diffusion term
        if diffusion > 0:
            laplacian = jax.scipy.signal.convolve2d(
                field, self.diffusion_kernel, mode='same', boundary='wrap'
            )
            diffusion_term = diffusion * laplacian
        else:
            diffusion_term = 0.0
        
        # Update field
        field_new = field + dt * (
            growth_amplitude * growth +  # Lenia dynamics
            diffusion_term +              # Extra diffusion
            injection_mask -              # External injection
            decay * field                 # Decay
        )
        
        # Clamp to valid range
        field_new = jnp.clip(field_new, 0.0, 1.0)
        
        return field_new
    
    def step(self, 
             positions: Optional[np.ndarray] = None,
             powers: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advance simulation by one step.
        
        Args:
            positions: (N, 2) array of (x, y) injection positions
            powers: (N,) array of power per position (or None for default)
            
        Returns:
            Current field as numpy array
        """
        # Create injection mask
        if positions is not None and len(positions) > 0:
            positions = jnp.array(positions, dtype=jnp.float32)
            if powers is None:
                powers = jnp.ones(len(positions)) * self.config.injection_power
            else:
                powers = jnp.array(powers, dtype=jnp.float32)
            
            injection_mask = create_injection_mask(
                positions, powers,
                (self.config.height, self.config.width),
                self.config.injection_radius
            )
        else:
            injection_mask = jnp.zeros_like(self.field)
        
        # Run step
        self.field = self._step_jit(
            self.field,
            injection_mask,
            self.lenia_kernel,
            self.config.dt,
            self.config.diffusion,
            self.config.decay,
            self.config.growth_mu,
            self.config.growth_sigma,
            self.config.growth_amplitude
        )
        
        return np.array(self.field)
    
    def get_field(self) -> np.ndarray:
        """Get current field as numpy array."""
        return np.array(self.field)
    
    def get_field_uint8(self) -> np.ndarray:
        """Get current field as uint8 image (0-255)."""
        return (np.array(self.field) * 255).astype(np.uint8)
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Rebuild kernel if needed
        if any(k in kwargs for k in ['kernel_radius', 'kernel_sigma']):
            self.lenia_kernel = make_kernel(self.config)


# Quick test
if __name__ == "__main__":
    import time
    
    config = FieldConfig(width=256, height=256)
    field = LeniaField(config)
    
    # Warmup
    for _ in range(10):
        field.step(np.array([[128, 128]]))
    
    # Benchmark
    n_steps = 100
    start = time.time()
    for _ in range(n_steps):
        field.step(np.array([[128, 128], [64, 64], [192, 192]]))
    elapsed = time.time() - start
    
    print(f"Resolution: {config.width}x{config.height}")
    print(f"FPS: {n_steps / elapsed:.1f}")
    print(f"Field max: {field.get_field().max():.3f}")
