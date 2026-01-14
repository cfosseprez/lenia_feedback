"""
Pytest fixtures for lenia_field tests.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lenia_field.core import FieldConfig, LeniaField


@pytest.fixture
def small_config():
    """Small field config for fast tests (classic injection mode).

    Note: injection_power must be > 1.0 to overcome negative Lenia growth
    when field is empty (growth = -1 when potential = 0).
    Uses suzuki_style=False for classic injection behavior.
    """
    return FieldConfig(
        width=64,
        height=64,
        kernel_radius=5,
        dt=0.1,
        diffusion=0.01,
        decay=0.001,
        injection_power=2.0,  # High enough to overcome growth=-1
        suzuki_style=False    # Use classic injection mode for these tests
    )


@pytest.fixture
def small_config_suzuki():
    """Small field config with Suzuki resource dynamics enabled."""
    return FieldConfig(
        width=64,
        height=64,
        kernel_radius=5,
        dt=0.1,
        diffusion=0.01,
        decay=0.001,
        suzuki_style=True,
        r_max=1.0,
        r_initial=1.0,
        R_C=0.005,
        R_G=0.001,
        resource_effect_rate=0.1
    )


@pytest.fixture
def default_config():
    """Default field config (with Suzuki mode enabled by default)."""
    return FieldConfig()


@pytest.fixture
def small_field(small_config):
    """Small LeniaField instance."""
    return LeniaField(small_config)


@pytest.fixture
def sample_positions():
    """Sample injection positions."""
    return np.array([
        [32, 32],
        [16, 16],
        [48, 48]
    ], dtype=np.float32)


@pytest.fixture
def sample_powers():
    """Sample injection powers.

    Must be > 1.0 to overcome negative Lenia growth.
    """
    return np.array([2.0, 2.5, 2.0], dtype=np.float32)


@pytest.fixture
def random_positions():
    """Random positions within 64x64 field."""
    np.random.seed(42)
    return np.random.rand(10, 2) * 64


@pytest.fixture
def edge_positions():
    """Positions at field edges."""
    return np.array([
        [0, 0],
        [63, 0],
        [0, 63],
        [63, 63],
        [32, 0],
        [0, 32]
    ], dtype=np.float32)
