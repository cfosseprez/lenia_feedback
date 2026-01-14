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
    growth_mu: float = 0.15    # Standard Lenia value (center of growth function)
    growth_sigma: float = 0.015  # Standard Lenia value (width of growth function)
    growth_amplitude: float = 1.0

    # Dynamics
    dt: float = 0.1           # Time step
    diffusion: float = 0.01   # Additional diffusion (beyond Lenia kernel)
    decay: float = 0.01       # Passive decay rate (lower to let patterns persist)

    # Morphogen injection (classic mode)
    injection_radius: float = 5.0   # Radius of injection spot
    injection_power: float = 0.1    # Intensity per injection point

    # Performance
    use_fft: bool = True      # Use FFT convolution (faster for large kernels)
    skip_diffusion: bool = True   # Skip extra diffusion convolution (not part of standard Lenia)

    # Suzuki-style resource dynamics (Suzuki et al., ALIFE 2023)
    # When enabled, agents affect resources AND inject morphogen.
    # Growth is modulated by local resource availability.
    suzuki_style: bool = True     # Enable resource-based dynamics (default: True)
    r_max: float = 1.0            # Maximum resource per cell
    r_initial: float = 1.0        # Initial resource level
    R_C: float = 0.005            # Paper value: resource consumption by morphogen
    R_G: float = 0.002            # Paper value: resource recovery rate towards r_max
    resource_rule: str = "S"      # "S" (state modulation) or "E" (energy modulation)
    agents_emit_resources: bool = True  # True: agents ADD resources, False: agents CONSUME
    resource_effect_rate: float = 0.5   # Multiplier for agent resource effect (emit or consume)
    resource_diffusion: float = 0.05    # Resource diffusion rate (spreads resources spatially)


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


@partial(jit, static_argnums=(2, 3))
def create_injection_mask_dense(positions: jnp.ndarray,
                                powers: jnp.ndarray,
                                shape: Tuple[int, int],
                                radius: float) -> jnp.ndarray:
    """Create injection mask from positions (dense version - slow for large fields).

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


def create_injection_mask_sparse(positions: np.ndarray,
                                 powers: np.ndarray,
                                 shape: Tuple[int, int],
                                 radius: float) -> np.ndarray:
    """Create injection mask using sparse local computation.

    MUCH faster than dense version for large fields (e.g., 2000x2000).
    Only computes Gaussian within a small bounding box around each agent.

    Complexity: O(N * box_size^2) instead of O(N * H * W)
    For 100 agents, radius=5: O(100 * 900) = 90K ops vs O(100 * 4M) = 400M ops

    Args:
        positions: (N, 2) array of (x, y) positions
        powers: (N,) array of injection powers per point
        shape: (height, width) of field
        radius: injection spot radius

    Returns:
        Injection mask as numpy array
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)

    if len(positions) == 0:
        return mask

    # Bounding box size (3 sigma covers 99.7% of Gaussian)
    box_size = int(radius * 3) + 1

    # Pre-compute local coordinate grid for the stamp
    local_coords = np.arange(-box_size, box_size + 1, dtype=np.float32)
    local_xx, local_yy = np.meshgrid(local_coords, local_coords)
    dist_sq_template = local_xx**2 + local_yy**2

    # Pre-compute Gaussian template (normalized, will be scaled by power)
    gaussian_template = np.exp(-dist_sq_template / (2 * radius**2))

    stamp_size = 2 * box_size + 1

    for i in range(len(positions)):
        x, y = positions[i]
        power = powers[i]

        if power <= 0:
            continue

        # Integer position for indexing
        ix, iy = int(round(x)), int(round(y))

        # Compute bounds with clipping
        x0 = ix - box_size
        y0 = iy - box_size
        x1 = ix + box_size + 1
        y1 = iy + box_size + 1

        # Clip to field boundaries
        field_x0 = max(0, x0)
        field_y0 = max(0, y0)
        field_x1 = min(w, x1)
        field_y1 = min(h, y1)

        # Corresponding stamp region
        stamp_x0 = field_x0 - x0
        stamp_y0 = field_y0 - y0
        stamp_x1 = stamp_size - (x1 - field_x1)
        stamp_y1 = stamp_size - (y1 - field_y1)

        # Add scaled Gaussian stamp
        if field_x1 > field_x0 and field_y1 > field_y0:
            mask[field_y0:field_y1, field_x0:field_x1] += (
                power * gaussian_template[stamp_y0:stamp_y1, stamp_x0:stamp_x1]
            )

    return mask


def create_injection_mask(positions, powers, shape, radius, use_sparse=True):
    """Create injection mask - auto-selects sparse or dense based on field size.

    Args:
        positions: (N, 2) array of (x, y) positions
        powers: (N,) array of injection powers per point
        shape: (height, width) of field
        radius: injection spot radius
        use_sparse: Force sparse mode (default True, much faster for large fields)
    """
    h, w = shape

    # Use sparse for large fields (> 500x500) or when explicitly requested
    if use_sparse or (h * w > 250000):
        positions_np = np.asarray(positions, dtype=np.float32)
        powers_np = np.asarray(powers, dtype=np.float32)
        return create_injection_mask_sparse(positions_np, powers_np, shape, radius)
    else:
        return create_injection_mask_dense(positions, powers, shape, radius)


class LeniaField:
    """Main class for running the morphogen field simulation.

    Supports two modes:
    - Classic mode: Agents inject morphogen directly into the field
    - Suzuki mode (default): Agents consume resources, growth modulated by resource availability
    """

    def __init__(self, config: Optional[FieldConfig] = None):
        self.config = config or FieldConfig()
        self.reset()

        # Pre-compile kernels
        self.lenia_kernel = make_kernel(self.config)
        self.diffusion_kernel = make_diffusion_kernel()

        # JIT compile step functions
        # Classic mode: skip_diffusion is static arg index 10
        self._step_classic_jit = jit(self._step_classic_impl, static_argnums=(10,))
        # Suzuki mode: skip_diffusion, resource_rule, agents_emit_resources are static
        # Args: field, resource, injection_mask, resource_effect_mask, lenia_kernel, diffusion_kernel,
        #       dt(6), diffusion(7), decay(8), growth_mu(9), growth_sigma(10), growth_amplitude(11),
        #       r_max(12), R_G(13), R_C(14), resource_diffusion(15), skip_diffusion(16), resource_rule(17), agents_emit_resources(18)
        self._step_suzuki_jit = jit(self._step_suzuki_impl, static_argnums=(16, 17, 18))

    def reset(self):
        """Reset field and resource to initial state."""
        self.field = jnp.zeros((self.config.height, self.config.width))
        # Initialize resource field for Suzuki mode
        self.resource = jnp.full(
            (self.config.height, self.config.width),
            self.config.r_initial,
            dtype=jnp.float32
        )

    def set_field(self, field: np.ndarray):
        """Set field from numpy array."""
        self.field = jnp.array(field)

    def set_resource(self, resource: np.ndarray):
        """Set resource field from numpy array."""
        self.resource = jnp.array(resource)
        
    def _step_classic_impl(self, field: jnp.ndarray,
                           injection_mask: jnp.ndarray,
                           lenia_kernel: jnp.ndarray,
                           diffusion_kernel: jnp.ndarray,
                           dt: float,
                           diffusion: float,
                           decay: float,
                           growth_mu: float,
                           growth_sigma: float,
                           growth_amplitude: float,
                           skip_diffusion: bool) -> jnp.ndarray:
        """Single step of the simulation (classic mode with direct injection)."""

        h, w = field.shape
        kh, kw = lenia_kernel.shape

        # Lenia-style update: convolve then apply growth function
        potential = convolve_fft(field, lenia_kernel, (h, w), (kh, kw))

        # Growth function
        growth = growth_function(potential, growth_mu, growth_sigma)

        # Only mask NEGATIVE growth in empty areas.
        # - Positive growth allowed everywhere (patterns can spread/evolve)
        # - Negative growth suppressed where field is empty (prevents injection cancelation)
        presence_mask = jnp.tanh(field * 10.0)  # 0 when field=0, ~1 when field>0.1
        masked_growth = jnp.where(growth > 0, growth, growth * presence_mask)

        # Additional diffusion term (optional - skip for large fields)
        if skip_diffusion:
            diffusion_term = 0.0
        else:
            dkh, dkw = diffusion_kernel.shape
            laplacian = convolve_fft(field, diffusion_kernel, (h, w), (dkh, dkw))
            diffusion_term = diffusion * laplacian

        # Update field
        field_new = field + dt * (
            growth_amplitude * masked_growth +  # Lenia dynamics (masked)
            diffusion_term +              # Extra diffusion
            injection_mask -              # External injection
            decay * field                 # Decay
        )

        # Clamp to valid range
        field_new = jnp.clip(field_new, 0.0, 1.0)

        return field_new

    def _step_suzuki_impl(self, field: jnp.ndarray,
                          resource: jnp.ndarray,
                          injection_mask: jnp.ndarray,
                          resource_effect_mask: jnp.ndarray,
                          lenia_kernel: jnp.ndarray,
                          diffusion_kernel: jnp.ndarray,
                          dt: float,
                          diffusion: float,
                          decay: float,
                          growth_mu: float,
                          growth_sigma: float,
                          growth_amplitude: float,
                          r_max: float,
                          R_G: float,
                          R_C: float,
                          resource_diffusion: float,
                          skip_diffusion: bool,
                          resource_rule: str,
                          agents_emit_resources: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single step with hybrid Suzuki resource dynamics.

        Combines external injection with resource-modulated growth:
        - Agents inject morphogen into the field (like classic mode)
        - Agents EMIT resources (feed the Lenia patterns)
        - Lenia patterns consume resources where they exist
        - Growth is modulated by available resources

        Based on Suzuki et al. (ALIFE 2023) but adapted for external stimulus.
        """
        h, w = field.shape
        kh, kw = lenia_kernel.shape

        # --- Resource Update ---
        # Resources are:
        # 1. ADDED by agent presence (emission_mask) - agents "feed" the field
        # 2. CONSUMED by morphogen presence (R_C * field) - patterns eat resources
        # 3. Naturally recover towards r_max (recovery term)
        # 4. DIFFUSE spatially (spread out)

        # Recovery term: resources grow back towards r_max
        recovery = R_G * (r_max - resource)

        # LOCAL consumption by morphogen presence (element-wise)
        # Resources are consumed WHERE morphogen exists
        consumption_by_morphogen = R_C * field

        # Agent resource effect: positive (emit) or negative (consume) based on config
        # agents_emit_resources=True: agents ADD resources (feed Lenia)
        # agents_emit_resources=False: agents CONSUME resources (deplete)
        agent_effect = jnp.where(agents_emit_resources, resource_effect_mask, -resource_effect_mask)

        # Resource diffusion using simple 3x3 Laplacian (much cheaper than FFT)
        # Laplacian: sum of neighbors minus 4*center
        padded = jnp.pad(resource, 1, mode='wrap')  # Wrap around edges
        resource_laplacian = (
            padded[:-2, 1:-1] +  # top
            padded[2:, 1:-1] +   # bottom
            padded[1:-1, :-2] +  # left
            padded[1:-1, 2:] -   # right
            4.0 * resource       # center
        )
        resource_diffusion_term = resource_diffusion * resource_laplacian

        # Update resource
        resource_new = resource + recovery - consumption_by_morphogen + agent_effect + resource_diffusion_term
        resource_new = jnp.clip(resource_new, 0.0, r_max)

        # --- Lenia Update with Resource Modulation ---
        # Growth uses resource_new - patterns are suppressed where resources are depleted

        # Only mask NEGATIVE growth in empty areas (allows positive growth for pattern evolution)
        presence_mask = jnp.tanh(field * 10.0)  # 0 when field=0, ~1 when field>0.1

        if resource_rule == "E":
            # Rule E: Multiply state by resource BEFORE convolution
            modulated_field = field * resource_new
            potential = convolve_fft(modulated_field, lenia_kernel, (h, w), (kh, kw))
            growth = growth_function(potential, growth_mu, growth_sigma)
            # Only mask negative growth
            masked_growth = jnp.where(growth > 0, growth, growth * presence_mask)
            modulated_growth = masked_growth
        else:
            # Rule S: Multiply growth by resource AFTER convolution
            potential = convolve_fft(field, lenia_kernel, (h, w), (kh, kw))
            growth = growth_function(potential, growth_mu, growth_sigma)
            # Only mask negative growth, then modulate by resource
            masked_growth = jnp.where(growth > 0, growth, growth * presence_mask)
            modulated_growth = masked_growth * resource_new

        # Additional diffusion term (optional)
        if skip_diffusion:
            diffusion_term = 0.0
        else:
            dkh, dkw = diffusion_kernel.shape
            laplacian = convolve_fft(field, diffusion_kernel, (h, w), (dkh, dkw))
            diffusion_term = diffusion * laplacian

        # Update field
        # Key: Injection is NOT modulated - agents always inject morphogen
        # Only growth is modulated by resources - patterns can't self-sustain in depleted areas
        field_new = field + dt * (
            growth_amplitude * modulated_growth +  # Resource-modulated Lenia dynamics
            diffusion_term +                       # Extra diffusion
            injection_mask -                       # Direct injection (not resource-modulated)
            decay * field                          # Decay
        )

        # Clamp to valid range
        field_new = jnp.clip(field_new, 0.0, 1.0)

        return field_new, resource_new
    
    def step(self,
             positions: Optional[np.ndarray] = None,
             powers: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advance simulation by one step.

        Args:
            positions: (N, 2) array of (x, y) positions
                - Classic mode: injection positions
                - Suzuki mode: agent positions (consume resources)
            powers: (N,) array of power per position (or None for default)

        Returns:
            Current field as numpy array
        """
        if self.config.suzuki_style:
            return self._step_suzuki(positions, powers)
        else:
            return self._step_classic(positions, powers)

    def _step_classic(self,
                      positions: Optional[np.ndarray] = None,
                      powers: Optional[np.ndarray] = None) -> np.ndarray:
        """Classic mode: agents inject morphogen directly."""
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
        self.field = self._step_classic_jit(
            self.field,
            injection_mask,
            self.lenia_kernel,
            self.diffusion_kernel,
            self.config.dt,
            self.config.diffusion,
            self.config.decay,
            self.config.growth_mu,
            self.config.growth_sigma,
            self.config.growth_amplitude,
            self.config.skip_diffusion
        )

        return np.array(self.field)

    def _step_suzuki(self,
                     positions: Optional[np.ndarray] = None,
                     powers: Optional[np.ndarray] = None) -> np.ndarray:
        """Suzuki mode: agents inject morphogen AND consume resources, growth modulated by resource.

        This is a hybrid approach that combines:
        - External injection (agents create morphogen)
        - Resource consumption (agents deplete local resources)
        - Growth modulation (Lenia growth scaled by available resource)

        This prevents runaway accumulation while still allowing agents to create patterns.
        """
        # Create injection mask (for adding morphogen) and consumption mask (for depleting resources)
        if positions is not None and len(positions) > 0:
            positions = jnp.array(positions, dtype=jnp.float32)

            # Injection powers (for adding morphogen to field)
            if powers is None:
                injection_powers = jnp.ones(len(positions)) * self.config.injection_power
            else:
                injection_powers = jnp.array(powers, dtype=jnp.float32)

            injection_mask = create_injection_mask(
                positions, injection_powers,
                (self.config.height, self.config.width),
                self.config.injection_radius
            )

            # Resource effect mask (for emit or consume)
            # Scale by injection power - higher injection = stronger resource effect
            resource_powers = injection_powers * self.config.resource_effect_rate
            resource_effect_mask = create_injection_mask(
                positions, resource_powers,
                (self.config.height, self.config.width),
                self.config.injection_radius
            )
        else:
            injection_mask = jnp.zeros_like(self.field)
            resource_effect_mask = jnp.zeros_like(self.field)

        # Run Suzuki step with injection
        self.field, self.resource = self._step_suzuki_jit(
            self.field,
            self.resource,
            injection_mask,
            resource_effect_mask,
            self.lenia_kernel,
            self.diffusion_kernel,
            self.config.dt,
            self.config.diffusion,
            self.config.decay,
            self.config.growth_mu,
            self.config.growth_sigma,
            self.config.growth_amplitude,
            self.config.r_max,
            self.config.R_G,
            self.config.R_C,
            self.config.resource_diffusion,
            self.config.skip_diffusion,
            self.config.resource_rule,
            self.config.agents_emit_resources
        )

        return np.array(self.field)
    
    def get_field(self) -> np.ndarray:
        """Get current field as numpy array."""
        return np.array(self.field)

    def get_field_uint8(self) -> np.ndarray:
        """Get current field as uint8 image (0-255)."""
        return (np.array(self.field) * 255).astype(np.uint8)

    def get_resource(self) -> np.ndarray:
        """Get current resource field as numpy array (Suzuki mode)."""
        return np.array(self.resource)

    def get_resource_uint8(self) -> np.ndarray:
        """Get current resource field as uint8 image (0-255), scaled by r_max."""
        normalized = np.array(self.resource) / self.config.r_max
        return (normalized * 255).astype(np.uint8)

    def step_with_metrics(self,
                          positions: Optional[np.ndarray] = None,
                          powers: Optional[np.ndarray] = None) -> dict:
        """
        Advance simulation by one step and return detailed metrics.

        Use this for debugging/analysis of Lenia dynamics.

        Returns:
            Dictionary with metrics:
                - field: current field array
                - field_mean, field_max, field_min, field_std: field statistics
                - resource_mean, resource_max, resource_min: resource statistics (Suzuki mode)
                - injection_total: total injection added this step
                - injection_max: max injection value
                - consumption_total: total resource consumed (Suzuki mode)
                - growth_positive: sum of positive growth contributions
                - growth_negative: sum of negative growth contributions
                - decay_total: total decay applied
                - delta_field: change in field from previous step
        """
        field_before = np.array(self.field)
        resource_before = np.array(self.resource) if self.config.suzuki_style else None

        # Compute masks for metrics
        if positions is not None and len(positions) > 0:
            positions_jnp = jnp.array(positions, dtype=jnp.float32)
            if powers is None:
                injection_powers = jnp.ones(len(positions_jnp)) * self.config.injection_power
            else:
                injection_powers = jnp.array(powers, dtype=jnp.float32)

            injection_mask = create_injection_mask(
                positions_jnp, injection_powers,
                (self.config.height, self.config.width),
                self.config.injection_radius
            )
            injection_mask_np = np.array(injection_mask)

            if self.config.suzuki_style:
                resource_powers = injection_powers * self.config.resource_effect_rate
                resource_effect_mask = create_injection_mask(
                    positions_jnp, resource_powers,
                    (self.config.height, self.config.width),
                    self.config.injection_radius
                )
                resource_effect_mask_np = np.array(resource_effect_mask)
            else:
                resource_effect_mask_np = np.zeros_like(field_before)
        else:
            injection_mask_np = np.zeros_like(field_before)
            resource_effect_mask_np = np.zeros_like(field_before)

        # Compute growth for metrics (before step)
        potential = np.array(convolve_fft(
            self.field, self.lenia_kernel,
            (self.config.height, self.config.width),
            self.lenia_kernel.shape
        ))
        growth = 2.0 * np.exp(-((potential - self.config.growth_mu) ** 2) /
                              (2 * self.config.growth_sigma ** 2)) - 1.0

        if self.config.suzuki_style:
            modulated_growth = growth * np.array(self.resource)
        else:
            modulated_growth = growth

        # Now do the actual step
        field_result = self.step(positions, powers)

        field_after = np.array(self.field)
        resource_after = np.array(self.resource) if self.config.suzuki_style else None

        # Compute metrics
        metrics = {
            # Field output
            'field': field_result,

            # Field statistics
            'field_mean': float(np.mean(field_after)),
            'field_max': float(np.max(field_after)),
            'field_min': float(np.min(field_after)),
            'field_std': float(np.std(field_after)),

            # Injection metrics
            'injection_total': float(np.sum(injection_mask_np)),
            'injection_max': float(np.max(injection_mask_np)),
            'injection_mean': float(np.mean(injection_mask_np)),

            # Growth metrics
            'growth_positive': float(np.sum(np.maximum(modulated_growth, 0))),
            'growth_negative': float(np.sum(np.minimum(modulated_growth, 0))),
            'growth_mean': float(np.mean(modulated_growth)),
            'growth_max': float(np.max(modulated_growth)),
            'growth_min': float(np.min(modulated_growth)),

            # Decay metrics
            'decay_total': float(np.sum(self.config.decay * field_before)),

            # Field change
            'delta_field_mean': float(np.mean(field_after - field_before)),
            'delta_field_max': float(np.max(field_after - field_before)),
            'delta_field_min': float(np.min(field_after - field_before)),
        }

        # Suzuki-specific metrics
        if self.config.suzuki_style:
            metrics.update({
                'resource_mean': float(np.mean(resource_after)),
                'resource_max': float(np.max(resource_after)),
                'resource_min': float(np.min(resource_after)),
                'resource_effect_total': float(np.sum(resource_effect_mask_np)),
                'resource_effect_max': float(np.max(resource_effect_mask_np)),
                'agents_emit_resources': self.config.agents_emit_resources,
                'delta_resource_mean': float(np.mean(resource_after - resource_before)),
            })

        return metrics

    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Rebuild kernel if needed
        if any(k in kwargs for k in ['kernel_radius', 'kernel_sigma']):
            self.lenia_kernel = make_kernel(self.config)

        # Reset resource if r_initial changed
        if 'r_initial' in kwargs:
            self.resource = jnp.full(
                (self.config.height, self.config.width),
                self.config.r_initial,
                dtype=jnp.float32
            )


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
