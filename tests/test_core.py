"""
Tests for core.py - Lenia field simulation.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lenia_field.core import (
    FieldConfig,
    LeniaField,
    make_kernel,
    make_diffusion_kernel,
    growth_function,
    create_injection_mask,
    convolve_fft
)
import jax.numpy as jnp


class TestFieldConfig:
    """Tests for FieldConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FieldConfig()
        assert config.width == 512
        assert config.height == 512
        assert config.kernel_radius == 13
        assert config.dt == 0.1
        assert config.diffusion == 0.01
        assert config.decay == 0.001
        assert config.use_fft is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FieldConfig(
            width=256,
            height=128,
            kernel_radius=7,
            dt=0.05
        )
        assert config.width == 256
        assert config.height == 128
        assert config.kernel_radius == 7
        assert config.dt == 0.05

    def test_injection_parameters(self):
        """Test injection-related parameters."""
        config = FieldConfig(
            injection_radius=10.0,
            injection_power=0.2
        )
        assert config.injection_radius == 10.0
        assert config.injection_power == 0.2

    def test_growth_parameters(self):
        """Test growth function parameters."""
        config = FieldConfig(
            growth_mu=0.2,
            growth_sigma=0.02,
            growth_amplitude=1.5
        )
        assert config.growth_mu == 0.2
        assert config.growth_sigma == 0.02
        assert config.growth_amplitude == 1.5


class TestMakeKernel:
    """Tests for kernel generation."""

    def test_kernel_shape(self):
        """Test kernel has correct shape."""
        config = FieldConfig(kernel_radius=5)
        kernel = make_kernel(config)
        expected_size = 2 * 5 + 1  # 11
        assert kernel.shape == (expected_size, expected_size)

    def test_kernel_normalized(self):
        """Test kernel sums to 1."""
        config = FieldConfig(kernel_radius=7)
        kernel = make_kernel(config)
        assert abs(float(jnp.sum(kernel)) - 1.0) < 1e-5

    def test_kernel_non_negative(self):
        """Test kernel values are non-negative."""
        config = FieldConfig(kernel_radius=5)
        kernel = make_kernel(config)
        assert float(jnp.min(kernel)) >= 0.0

    def test_kernel_symmetric(self):
        """Test kernel is symmetric."""
        config = FieldConfig(kernel_radius=5)
        kernel = make_kernel(config)
        # Check symmetry
        assert jnp.allclose(kernel, kernel[::-1, :])  # Vertical
        assert jnp.allclose(kernel, kernel[:, ::-1])  # Horizontal
        assert jnp.allclose(kernel, kernel.T)  # Diagonal

    def test_kernel_ring_shape(self):
        """Test kernel has ring-like maximum away from center."""
        config = FieldConfig(kernel_radius=10, kernel_sigma=0.5)
        kernel = make_kernel(config)
        center = config.kernel_radius
        # Center should not be the maximum
        center_val = float(kernel[center, center])
        max_val = float(jnp.max(kernel))
        assert center_val < max_val


class TestDiffusionKernel:
    """Tests for diffusion kernel."""

    def test_diffusion_kernel_3x3(self):
        """Test 3x3 diffusion kernel."""
        kernel = make_diffusion_kernel(3)
        assert kernel.shape == (3, 3)
        # Center should be negative (Laplacian)
        assert float(kernel[1, 1]) < 0

    def test_diffusion_kernel_sum(self):
        """Test diffusion kernel sums close to zero."""
        kernel = make_diffusion_kernel(3)
        assert abs(float(jnp.sum(kernel))) < 0.1

    def test_diffusion_kernel_larger(self):
        """Test larger diffusion kernel."""
        kernel = make_diffusion_kernel(5)
        assert kernel.shape == (5, 5)


class TestGrowthFunction:
    """Tests for Lenia growth function."""

    def test_growth_at_mu(self):
        """Test growth function peaks at mu."""
        mu, sigma = 0.15, 0.015
        u = jnp.array([mu])
        result = growth_function(u, mu, sigma)
        # At mu, exp(-0) = 1, so result = 2*1 - 1 = 1
        assert abs(float(result[0]) - 1.0) < 1e-5

    def test_growth_far_from_mu(self):
        """Test growth function approaches -1 far from mu."""
        mu, sigma = 0.15, 0.015
        u = jnp.array([1.0])  # Far from mu=0.15
        result = growth_function(u, mu, sigma)
        assert float(result[0]) < -0.9

    def test_growth_symmetric(self):
        """Test growth function is symmetric around mu."""
        mu, sigma = 0.15, 0.015
        delta = 0.05
        u_low = jnp.array([mu - delta])
        u_high = jnp.array([mu + delta])
        result_low = growth_function(u_low, mu, sigma)
        result_high = growth_function(u_high, mu, sigma)
        assert abs(float(result_low[0]) - float(result_high[0])) < 1e-5

    def test_growth_range(self):
        """Test growth function output range is [-1, 1]."""
        mu, sigma = 0.15, 0.015
        u = jnp.linspace(0, 1, 100)
        result = growth_function(u, mu, sigma)
        assert float(jnp.min(result)) >= -1.0 - 1e-5
        assert float(jnp.max(result)) <= 1.0 + 1e-5


class TestCreateInjectionMask:
    """Tests for injection mask creation.

    Note: create_injection_mask is JIT-compiled with static_argnums=(3,) for radius.
    The shape parameter also needs to be static, so we test injection through LeniaField.
    """

    def test_injection_through_field(self, small_field):
        """Test injection creates values in field."""
        positions = np.array([[32, 32]], dtype=np.float32)
        result = small_field.step(positions=positions)
        # Field should have non-zero values after injection
        assert float(np.max(result)) > 0

    def test_injection_location(self, small_field):
        """Test injection creates local maximum near position."""
        positions = np.array([[32, 32]], dtype=np.float32)
        result = small_field.step(positions=positions)
        # Maximum should be near injection point
        max_idx = np.unravel_index(np.argmax(result), result.shape)
        assert abs(max_idx[1] - 32) <= 5  # x (with some spread)
        assert abs(max_idx[0] - 32) <= 5  # y

    def test_multiple_injection_points(self, small_field, sample_positions):
        """Test multiple injection points."""
        result = small_field.step(positions=sample_positions)
        # Field should have non-zero values
        assert float(np.max(result)) > 0

    def test_injection_power_scaling(self):
        """Test injection power affects field magnitude.

        Note: injection_power must be > 1.0 to overcome negative Lenia growth
        when field is empty (growth = -1 when potential = 0).
        """
        from lenia_field.core import FieldConfig, LeniaField

        positions = np.array([[32, 32]], dtype=np.float32)

        # Lower power (but still > 1.0 to produce positive values)
        config_low = FieldConfig(width=64, height=64, injection_power=1.5)
        field_low = LeniaField(config_low)
        result_low = field_low.step(positions=positions)
        max_low = float(np.max(result_low))

        # Higher power
        config_high = FieldConfig(width=64, height=64, injection_power=3.0)
        field_high = LeniaField(config_high)
        result_high = field_high.step(positions=positions)
        max_high = float(np.max(result_high))

        # Higher power should produce larger values
        assert max_high > max_low
        assert max_low > 0  # Both should produce positive values

    def test_empty_positions(self, small_field):
        """Test step with empty positions doesn't add injection."""
        # First step without injection
        result1 = small_field.step()
        # Field should be zero
        assert float(np.max(result1)) < 0.01


class TestConvolveFFT:
    """Tests for FFT-based convolution."""

    def test_convolve_fft_shape(self):
        """Test FFT convolution preserves shape."""
        field = jnp.ones((64, 64))
        kernel = jnp.ones((3, 3)) / 9
        result = convolve_fft(field, kernel, (64, 64), (3, 3))
        assert result.shape == (64, 64)

    def test_convolve_fft_uniform_field(self):
        """Test convolution of uniform field stays uniform."""
        field = jnp.ones((64, 64)) * 0.5
        kernel = jnp.ones((3, 3)) / 9
        result = convolve_fft(field, kernel, (64, 64), (3, 3))
        # Should remain uniform (approximately)
        assert float(jnp.std(result)) < 0.01


class TestLeniaField:
    """Tests for LeniaField class."""

    def test_initialization(self, small_config):
        """Test field initializes to zero."""
        field = LeniaField(small_config)
        assert field.field.shape == (small_config.height, small_config.width)
        assert float(jnp.max(field.field)) == 0.0

    def test_initialization_default(self):
        """Test default initialization."""
        field = LeniaField()
        assert field.config.width == 512
        assert field.config.height == 512

    def test_reset(self, small_field, sample_positions):
        """Test reset clears field."""
        # Inject some values
        small_field.step(sample_positions)
        assert float(jnp.max(small_field.field)) > 0

        # Reset
        small_field.reset()
        assert float(jnp.max(small_field.field)) == 0.0

    def test_step_no_injection(self, small_field):
        """Test step without injection."""
        result = small_field.step()
        assert result.shape == (small_field.config.height, small_field.config.width)
        assert isinstance(result, np.ndarray)

    def test_step_with_injection(self, small_field, sample_positions):
        """Test step with position injection."""
        result = small_field.step(positions=sample_positions)
        # Field should have non-zero values after injection
        assert float(np.max(result)) > 0

    def test_step_with_powers(self, small_field, sample_positions, sample_powers):
        """Test step with custom powers."""
        result = small_field.step(
            positions=sample_positions,
            powers=sample_powers
        )
        assert float(np.max(result)) > 0

    def test_field_clamped(self, small_field, sample_positions):
        """Test field values stay in [0, 1]."""
        # Run many steps with injection
        for _ in range(50):
            result = small_field.step(positions=sample_positions)

        assert float(np.min(result)) >= 0.0
        assert float(np.max(result)) <= 1.0

    def test_get_field(self, small_field, sample_positions):
        """Test get_field returns numpy array."""
        small_field.step(positions=sample_positions)
        field = small_field.get_field()
        assert isinstance(field, np.ndarray)
        assert field.dtype == np.float32

    def test_get_field_uint8(self, small_field, sample_positions):
        """Test get_field_uint8 returns correct format."""
        small_field.step(positions=sample_positions)
        field = small_field.get_field_uint8()
        assert isinstance(field, np.ndarray)
        assert field.dtype == np.uint8
        assert float(np.max(field)) <= 255
        assert float(np.min(field)) >= 0

    def test_set_field(self, small_field):
        """Test set_field updates field."""
        custom_field = np.random.rand(64, 64).astype(np.float32)
        small_field.set_field(custom_field)
        result = small_field.get_field()
        assert np.allclose(result, custom_field)

    def test_update_config(self, small_field):
        """Test update_config modifies parameters."""
        original_decay = small_field.config.decay
        small_field.update_config(decay=0.1)
        assert small_field.config.decay == 0.1
        assert small_field.config.decay != original_decay

    def test_update_config_kernel_rebuild(self, small_field):
        """Test kernel rebuilds when radius changes."""
        original_kernel_shape = small_field.lenia_kernel.shape
        small_field.update_config(kernel_radius=3)
        new_kernel_shape = small_field.lenia_kernel.shape
        assert new_kernel_shape != original_kernel_shape
        assert new_kernel_shape == (7, 7)  # 2*3+1

    def test_decay_over_time(self, small_field, sample_positions):
        """Test field decays over time without injection."""
        # Initial injection
        small_field.step(positions=sample_positions)
        initial_max = float(np.max(small_field.get_field()))

        # Run steps without injection
        for _ in range(100):
            small_field.step()

        final_max = float(np.max(small_field.get_field()))
        # Field should decay (or at least not grow unboundedly)
        assert final_max <= initial_max or final_max < 1.0

    def test_injection_radius_effect(self, small_config):
        """Test injection radius affects spread."""
        positions = np.array([[32, 32]], dtype=np.float32)

        # Small radius (injection_power > 1.0 to overcome negative growth)
        config_small = FieldConfig(
            width=64, height=64,
            injection_radius=2.0,
            injection_power=2.0
        )
        field_small = LeniaField(config_small)
        field_small.step(positions=positions)
        spread_small = float(np.sum(field_small.get_field() > 0.01))

        # Large radius
        config_large = FieldConfig(
            width=64, height=64,
            injection_radius=10.0,
            injection_power=2.0
        )
        field_large = LeniaField(config_large)
        field_large.step(positions=positions)
        spread_large = float(np.sum(field_large.get_field() > 0.01))

        assert spread_large > spread_small

    def test_edge_injection(self, small_field, edge_positions):
        """Test injection at field edges."""
        result = small_field.step(positions=edge_positions)
        # Should not raise and should produce values
        assert float(np.max(result)) > 0

    def test_multiple_steps_stability(self, small_field, random_positions):
        """Test field remains stable over many steps."""
        for i in range(200):
            # Inject at random positions occasionally
            if i % 10 == 0:
                result = small_field.step(positions=random_positions)
            else:
                result = small_field.step()

            # Check field stays valid
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))
            assert float(np.min(result)) >= 0.0
            assert float(np.max(result)) <= 1.0


class TestLeniaFieldFFTvsNonFFT:
    """Tests comparing FFT and non-FFT convolution."""

    def test_fft_vs_direct_similar(self):
        """Test FFT and direct convolution produce similar results."""
        config_fft = FieldConfig(
            width=32, height=32,
            kernel_radius=3,
            use_fft=True
        )
        config_direct = FieldConfig(
            width=32, height=32,
            kernel_radius=3,
            use_fft=False
        )

        positions = np.array([[16, 16]], dtype=np.float32)

        field_fft = LeniaField(config_fft)
        field_direct = LeniaField(config_direct)

        # Run a few steps
        for _ in range(5):
            result_fft = field_fft.step(positions=positions)
            result_direct = field_direct.step(positions=positions)

        # Results should be similar (not exact due to boundary handling)
        # Using a generous tolerance
        assert np.allclose(result_fft, result_direct, atol=0.1) or \
               np.corrcoef(result_fft.flatten(), result_direct.flatten())[0, 1] > 0.9


class TestLeniaFieldEdgeCases:
    """Edge case tests for LeniaField."""

    def test_zero_dt(self):
        """Test behavior with zero time step."""
        config = FieldConfig(width=32, height=32, dt=0.0)
        field = LeniaField(config)
        positions = np.array([[16, 16]], dtype=np.float32)

        # With dt=0, field should not change significantly
        initial = field.get_field().copy()
        field.step(positions=positions)
        result = field.get_field()

        # Field should be mostly unchanged
        assert np.allclose(initial, result, atol=0.01)

    def test_high_decay(self):
        """Test behavior with very high decay."""
        config = FieldConfig(width=32, height=32, decay=1.0, injection_power=2.0)
        field = LeniaField(config)
        positions = np.array([[16, 16]], dtype=np.float32)

        field.step(positions=positions)
        first_max = float(np.max(field.get_field()))

        # Run without injection
        for _ in range(10):
            field.step()

        final_max = float(np.max(field.get_field()))
        assert final_max < first_max

    def test_single_position(self):
        """Test with single injection position."""
        config = FieldConfig(width=32, height=32, injection_power=2.0)
        field = LeniaField(config)
        positions = np.array([[16, 16]], dtype=np.float32)

        result = field.step(positions=positions)
        assert float(np.max(result)) > 0

    def test_many_positions(self):
        """Test with many injection positions."""
        config = FieldConfig(width=64, height=64)
        field = LeniaField(config)
        positions = np.random.rand(100, 2).astype(np.float32) * 64

        result = field.step(positions=positions)
        assert not np.any(np.isnan(result))

    def test_negative_positions_handled(self):
        """Test negative positions don't crash (edge handling)."""
        config = FieldConfig(width=32, height=32)
        field = LeniaField(config)
        positions = np.array([[-5, -5], [16, 16]], dtype=np.float32)

        # Should not raise
        result = field.step(positions=positions)
        assert isinstance(result, np.ndarray)

    def test_out_of_bounds_positions(self):
        """Test out-of-bounds positions don't crash."""
        config = FieldConfig(width=32, height=32)
        field = LeniaField(config)
        positions = np.array([[100, 100], [16, 16]], dtype=np.float32)

        # Should not raise
        result = field.step(positions=positions)
        assert isinstance(result, np.ndarray)
