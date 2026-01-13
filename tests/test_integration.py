"""
Integration tests for lenia_field package.
These tests require the full stack: client spawning server subprocess.

Run with: pytest tests/test_integration.py -v -s

Note: These tests spawn real processes and use real network ports.
They may take longer and require ZMQ to be properly installed.
"""

import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lenia_field.client import LeniaClient
from lenia_field.core import LeniaField, FieldConfig


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def get_free_port():
    """Get a free port for testing."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class TestClientServerIntegration:
    """Full integration tests with real server subprocess."""

    @pytest.fixture
    def client(self):
        """Spawn a real client/server pair.

        Note: injection_power must be > 1.0 to overcome negative Lenia growth.
        """
        port = get_free_port()
        client = LeniaClient.spawn(
            port=port,
            width=64,
            height=64,
            dt=0.1,
            diffusion=0.01,
            decay=0.001,
            startup_timeout=10.0
        )
        # Set higher injection power to overcome negative growth
        client.config(injection_power=2.0)
        yield client
        try:
            client.shutdown()
        except:
            client.close()

    def test_spawn_and_ping(self, client):
        """Test spawning server and ping."""
        response = client.ping()
        assert response.ok is True

    def test_step_returns_field(self, client):
        """Test step returns valid field."""
        response = client.step()
        assert response.ok is True
        assert response.field is not None
        assert response.field.shape == (64, 64)
        assert response.field.dtype == np.float32

    def test_step_with_injection(self, client):
        """Test step with position injection."""
        positions = np.array([[32, 32], [16, 16]], dtype=np.float32)
        response = client.step(positions=positions)

        assert response.ok is True
        assert float(np.max(response.field)) > 0

    def test_step_with_custom_powers(self, client):
        """Test step with custom injection powers (must be > 1.0)."""
        positions = np.array([[32, 32]], dtype=np.float32)
        powers = np.array([2.5], dtype=np.float32)

        response = client.step(positions=positions, powers=powers)
        assert response.ok is True
        assert float(np.max(response.field)) > 0

    def test_multiple_steps(self, client):
        """Test multiple consecutive steps."""
        positions = np.array([[32, 32]], dtype=np.float32)

        for i in range(20):
            response = client.step(positions=positions)
            assert response.ok is True
            assert not np.any(np.isnan(response.field))

    def test_field_accumulates(self, client):
        """Test field values accumulate with injection."""
        positions = np.array([[32, 32]], dtype=np.float32)

        # First step
        response1 = client.step(positions=positions)
        max1 = float(np.max(response1.field))

        # More steps
        for _ in range(5):
            response = client.step(positions=positions)

        max_after = float(np.max(response.field))

        # Field should have accumulated (up to saturation)
        assert max_after >= max1

    def test_reset_clears_field(self, client):
        """Test reset clears the field."""
        positions = np.array([[32, 32]], dtype=np.float32)

        # Inject
        client.step(positions=positions)

        # Reset
        reset_response = client.reset()
        assert reset_response.ok is True

        # Get field
        get_response = client.get()
        assert float(np.max(get_response.field)) == 0.0

    def test_get_without_step(self, client):
        """Test get returns field without stepping."""
        positions = np.array([[32, 32]], dtype=np.float32)

        # Step to create some values
        client.step(positions=positions)

        # Get multiple times - field shouldn't change
        get1 = client.get()
        get2 = client.get()

        assert np.array_equal(get1.field, get2.field)

    def test_config_update(self, client):
        """Test config update works."""
        response = client.config(decay=0.1, diffusion=0.05)
        assert response.ok is True
        assert response.config['decay'] == 0.1
        assert response.config['diffusion'] == 0.05

    def test_uint8_output(self, client):
        """Test uint8 field output."""
        positions = np.array([[32, 32]], dtype=np.float32)

        response = client.step(positions=positions, uint8=True)
        assert response.ok is True
        assert response.field.dtype == np.uint8
        assert float(np.max(response.field)) <= 255

    def test_return_field_false(self, client):
        """Test step without returning field."""
        response = client.step(return_field=False)
        assert response.ok is True
        assert response.field is None

    def test_shutdown(self, client):
        """Test shutdown command."""
        response = client.shutdown()
        assert response.ok is True

        # Further commands should fail (either exception or error response)
        # After shutdown, the socket is closed so we can't make more requests
        # The fixture will handle cleanup


class TestClientServerStress:
    """Stress tests for client/server."""

    @pytest.fixture
    def client(self):
        """Spawn a real client/server pair."""
        port = get_free_port()
        client = LeniaClient.spawn(
            port=port,
            width=128,
            height=128,
            startup_timeout=10.0
        )
        yield client
        try:
            client.shutdown()
        except:
            client.close()

    def test_rapid_steps(self, client):
        """Test rapid consecutive steps."""
        positions = np.array([[64, 64]], dtype=np.float32)

        start = time.time()
        n_steps = 100

        for _ in range(n_steps):
            response = client.step(positions=positions)
            assert response.ok is True

        elapsed = time.time() - start
        fps = n_steps / elapsed

        print(f"\nRapid steps: {fps:.1f} FPS")
        assert fps > 10  # Should be able to do at least 10 FPS

    def test_many_injection_points(self, client):
        """Test injection with many points."""
        n_points = 50
        positions = np.random.rand(n_points, 2).astype(np.float32) * 128

        response = client.step(positions=positions)
        assert response.ok is True
        assert not np.any(np.isnan(response.field))

    def test_config_during_simulation(self, client):
        """Test changing config during simulation."""
        positions = np.array([[64, 64]], dtype=np.float32)

        for i in range(20):
            response = client.step(positions=positions)
            assert response.ok is True

            # Change config periodically
            if i % 5 == 0:
                client.config(decay=0.001 * (i + 1))


class TestCoreDirectUsage:
    """Tests for direct LeniaField usage without server."""

    def test_direct_field_creation(self):
        """Test creating field directly."""
        config = FieldConfig(width=64, height=64)
        field = LeniaField(config)

        result = field.step()
        assert result.shape == (64, 64)

    def test_direct_injection(self):
        """Test direct injection (injection_power > 1.0 needed)."""
        config = FieldConfig(width=64, height=64, injection_power=2.0)
        field = LeniaField(config)

        positions = np.array([[32, 32]], dtype=np.float32)
        result = field.step(positions=positions)

        assert float(np.max(result)) > 0

    def test_direct_many_steps(self):
        """Test many steps with direct field."""
        config = FieldConfig(width=64, height=64)
        field = LeniaField(config)

        positions = np.array([[32, 32]], dtype=np.float32)

        for _ in range(100):
            result = field.step(positions=positions)
            assert not np.any(np.isnan(result))
            assert float(np.min(result)) >= 0.0
            assert float(np.max(result)) <= 1.0

    def test_direct_performance(self):
        """Test direct field performance."""
        config = FieldConfig(width=128, height=128)
        field = LeniaField(config)

        positions = np.array([[64, 64]], dtype=np.float32)

        # Warmup
        for _ in range(10):
            field.step(positions=positions)

        # Benchmark
        start = time.time()
        n_steps = 100
        for _ in range(n_steps):
            field.step(positions=positions)
        elapsed = time.time() - start

        fps = n_steps / elapsed
        print(f"\nDirect field: {fps:.1f} FPS at 128x128")
        assert fps > 50  # Should be fast


class TestEdgeCasesIntegration:
    """Edge case integration tests."""

    @pytest.fixture
    def client(self):
        """Spawn a real client/server pair with high injection power."""
        port = get_free_port()
        client = LeniaClient.spawn(
            port=port,
            width=64,
            height=64,
            startup_timeout=10.0
        )
        client.config(injection_power=2.0)
        yield client
        try:
            client.shutdown()
        except:
            client.close()

    def test_empty_positions(self, client):
        """Test step with empty positions."""
        response = client.step(positions=np.array([]).reshape(0, 2))
        assert response.ok is True

    def test_list_positions(self, client):
        """Test step with list (not array) positions."""
        response = client.step(positions=[[32, 32], [16, 16]])
        assert response.ok is True
        assert float(np.max(response.field)) > 0

    def test_edge_positions(self, client):
        """Test injection at field edges."""
        positions = np.array([
            [0, 0],
            [63, 0],
            [0, 63],
            [63, 63]
        ], dtype=np.float32)

        response = client.step(positions=positions)
        assert response.ok is True
        assert not np.any(np.isnan(response.field))


class TestContextManager:
    """Tests for context manager usage."""

    def test_context_manager(self):
        """Test using client as context manager."""
        port = get_free_port()

        with LeniaClient.spawn(
            port=port,
            width=32,
            height=32,
            startup_timeout=10.0
        ) as client:
            response = client.ping()
            assert response.ok is True

            response = client.step()
            assert response.ok is True

        # After exiting, client should be closed
        # (server shutdown may have been called)


class TestExampleTracking:
    """Tests simulating tracking integration."""

    @pytest.fixture
    def client(self):
        """Spawn a real client/server pair."""
        port = get_free_port()
        client = LeniaClient.spawn(
            port=port,
            width=128,
            height=128,
            startup_timeout=10.0
        )
        yield client
        try:
            client.shutdown()
        except:
            client.close()

    def test_tracking_simulation(self, client):
        """Simulate tracking pipeline integration."""
        # Simulate tracked organisms
        n_organisms = 5
        positions = np.random.rand(n_organisms, 2).astype(np.float32) * 128
        velocities = (np.random.rand(n_organisms, 2) - 0.5) * 2

        for frame in range(50):
            # Update positions (simple physics)
            positions += velocities

            # Boundary reflection
            for i in range(2):
                mask_low = positions[:, i] < 0
                mask_high = positions[:, i] >= 128
                positions[mask_low, i] *= -1
                positions[mask_high, i] = 256 - positions[mask_high, i] - 1
                velocities[mask_low, i] *= -1
                velocities[mask_high, i] *= -1

            # Inject into field
            response = client.step(positions=positions)
            assert response.ok is True

            # Verify field is valid
            field = response.field
            assert not np.any(np.isnan(field))
            assert float(np.min(field)) >= 0.0
            assert float(np.max(field)) <= 1.0

    def test_variable_excretion(self, client):
        """Test variable excretion rates."""
        n_organisms = 5
        positions = np.random.rand(n_organisms, 2).astype(np.float32) * 128
        powers = np.random.rand(n_organisms).astype(np.float32) * 0.2

        for _ in range(20):
            response = client.step(positions=positions, powers=powers)
            assert response.ok is True

            # Vary powers
            powers = np.random.rand(n_organisms).astype(np.float32) * 0.2
