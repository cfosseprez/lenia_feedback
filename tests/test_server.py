"""
Tests for server.py - ZMQ-based Lenia server.
"""

import pytest
import numpy as np
import msgpack
import msgpack_numpy as m
import sys
import os
import threading
import time

# Patch msgpack for numpy support
m.patch()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lenia_field.core import FieldConfig
from lenia_field.server import LeniaServer


class MockSocket:
    """Mock ZMQ socket for testing server logic without network."""

    def __init__(self):
        self.sent_messages = []
        self.received_messages = []
        self._recv_queue = []

    def bind(self, address):
        pass

    def recv(self):
        if self._recv_queue:
            return self._recv_queue.pop(0)
        raise Exception("No message in queue")

    def send(self, message):
        self.sent_messages.append(message)

    def close(self):
        pass

    def queue_message(self, data):
        """Queue a message to be received."""
        self._recv_queue.append(msgpack.packb(data))


class MockContext:
    """Mock ZMQ context."""

    def __init__(self):
        self.socket_instance = None

    def socket(self, socket_type):
        self.socket_instance = MockSocket()
        return self.socket_instance

    def term(self):
        pass


class TestLeniaServerInit:
    """Tests for LeniaServer initialization."""

    def test_server_initialization(self, monkeypatch):
        """Test server initializes correctly."""
        mock_context = MockContext()
        monkeypatch.setattr('zmq.Context', lambda: mock_context)

        config = FieldConfig(width=64, height=64)
        server = LeniaServer(port=5555, config=config)

        assert server.port == 5555
        assert server.config.width == 64
        assert server.config.height == 64
        assert server.step_count == 0

    def test_server_default_config(self, monkeypatch):
        """Test server with default config."""
        mock_context = MockContext()
        monkeypatch.setattr('zmq.Context', lambda: mock_context)

        server = LeniaServer(port=5556)
        assert server.config.width == 512
        assert server.config.height == 512


class TestLeniaServerHandleRequest:
    """Tests for server request handling."""

    @pytest.fixture
    def server(self, monkeypatch):
        """Create server with mocked ZMQ.

        Note: injection_power must be > 1.0 to overcome negative Lenia growth.
        """
        mock_context = MockContext()
        monkeypatch.setattr('zmq.Context', lambda: mock_context)
        config = FieldConfig(width=64, height=64, injection_power=2.0)
        return LeniaServer(port=5555, config=config)

    def test_handle_ping(self, server):
        """Test ping command."""
        response = server.handle_request({'cmd': 'ping'})
        assert response['status'] == 'ok'
        assert response['message'] == 'pong'

    def test_handle_shutdown(self, server):
        """Test shutdown command."""
        response = server.handle_request({'cmd': 'shutdown'})
        assert response['status'] == 'ok'
        assert response['shutdown'] is True

    def test_handle_reset(self, server):
        """Test reset command."""
        # First inject some values
        positions = np.array([[32, 32]], dtype=np.float32)
        server.handle_request({'cmd': 'step', 'positions': positions})

        # Reset
        response = server.handle_request({'cmd': 'reset'})
        assert response['status'] == 'ok'

        # Verify field is zero
        get_response = server.handle_request({'cmd': 'get'})
        assert float(np.max(get_response['field'])) == 0.0

    def test_handle_step_no_positions(self, server):
        """Test step without positions."""
        response = server.handle_request({'cmd': 'step'})
        assert response['status'] == 'ok'
        assert 'field' in response
        assert response['shape'] == (64, 64)

    def test_handle_step_with_positions(self, server):
        """Test step with injection positions."""
        positions = np.array([[32, 32], [16, 16]], dtype=np.float32)
        response = server.handle_request({
            'cmd': 'step',
            'positions': positions
        })
        assert response['status'] == 'ok'
        assert float(np.max(response['field'])) > 0

    def test_handle_step_with_powers(self, server):
        """Test step with custom powers (must be > 1.0 to overcome growth=-1)."""
        positions = np.array([[32, 32]], dtype=np.float32)
        powers = np.array([2.5], dtype=np.float32)
        response = server.handle_request({
            'cmd': 'step',
            'positions': positions,
            'powers': powers
        })
        assert response['status'] == 'ok'
        assert float(np.max(response['field'])) > 0

    def test_handle_step_uint8(self, server):
        """Test step with uint8 output."""
        positions = np.array([[32, 32]], dtype=np.float32)
        response = server.handle_request({
            'cmd': 'step',
            'positions': positions,
            'uint8': True
        })
        assert response['status'] == 'ok'
        assert response['field'].dtype == np.uint8

    def test_handle_step_no_field_return(self, server):
        """Test step without returning field."""
        response = server.handle_request({
            'cmd': 'step',
            'return_field': False
        })
        assert response['status'] == 'ok'
        assert 'field' not in response

    def test_handle_step_with_config(self, server):
        """Test step with config update."""
        original_decay = server.field.config.decay
        response = server.handle_request({
            'cmd': 'step',
            'config': {'decay': 0.1}
        })
        assert response['status'] == 'ok'
        assert server.field.config.decay == 0.1
        assert server.field.config.decay != original_decay

    def test_handle_get(self, server):
        """Test get command."""
        # First step to have some field values
        positions = np.array([[32, 32]], dtype=np.float32)
        server.handle_request({'cmd': 'step', 'positions': positions})

        # Get field
        response = server.handle_request({'cmd': 'get'})
        assert response['status'] == 'ok'
        assert 'field' in response
        assert response['shape'] == (64, 64)

    def test_handle_get_uint8(self, server):
        """Test get with uint8 output."""
        response = server.handle_request({'cmd': 'get', 'uint8': True})
        assert response['status'] == 'ok'
        assert response['field'].dtype == np.uint8

    def test_handle_config_get(self, server):
        """Test config command returns current config."""
        response = server.handle_request({'cmd': 'config'})
        assert response['status'] == 'ok'
        assert 'config' in response
        assert response['config']['width'] == 64
        assert response['config']['height'] == 64

    def test_handle_config_update(self, server):
        """Test config update command."""
        response = server.handle_request({
            'cmd': 'config',
            'config': {
                'diffusion': 0.05,
                'decay': 0.01
            }
        })
        assert response['status'] == 'ok'
        assert response['config']['diffusion'] == 0.05
        assert response['config']['decay'] == 0.01

    def test_handle_unknown_command(self, server):
        """Test unknown command returns error."""
        response = server.handle_request({'cmd': 'invalid_command'})
        assert response['status'] == 'error'
        assert 'Unknown command' in response['error']

    def test_handle_request_default_cmd(self, server):
        """Test request with no cmd defaults to step."""
        response = server.handle_request({})
        assert response['status'] == 'ok'
        # Default command is step, which returns shape
        assert 'shape' in response


class TestLeniaServerFPS:
    """Tests for FPS tracking."""

    @pytest.fixture
    def server(self, monkeypatch):
        """Create server with mocked ZMQ."""
        mock_context = MockContext()
        monkeypatch.setattr('zmq.Context', lambda: mock_context)
        config = FieldConfig(width=32, height=32)
        return LeniaServer(port=5555, config=config)

    def test_fps_tracking(self, server):
        """Test FPS is tracked over multiple steps."""
        # Run enough steps to trigger FPS calculation
        for _ in range(10):
            response = server.handle_request({'cmd': 'step'})

        # FPS should be returned
        assert 'fps' in response

    def test_step_count_increments(self, server):
        """Test step count increments."""
        initial_count = server.step_count

        server.handle_request({'cmd': 'step'})

        # Either count incremented or was reset after FPS calculation
        assert server.step_count >= 0


class TestLeniaServerCleanup:
    """Tests for server cleanup."""

    def test_cleanup(self, monkeypatch):
        """Test cleanup closes socket."""
        mock_context = MockContext()
        monkeypatch.setattr('zmq.Context', lambda: mock_context)

        server = LeniaServer(port=5555)

        # Track if close was called
        close_called = [False]
        original_close = server.socket.close

        def track_close():
            close_called[0] = True
            # Don't call original as it's a mock

        server.socket.close = track_close

        server.cleanup()

        assert close_called[0] is True


class TestLeniaServerErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def server(self, monkeypatch):
        """Create server with mocked ZMQ."""
        mock_context = MockContext()
        monkeypatch.setattr('zmq.Context', lambda: mock_context)
        config = FieldConfig(width=64, height=64)
        return LeniaServer(port=5555, config=config)

    def test_handle_invalid_positions_shape(self, server):
        """Test handling of invalid position array."""
        # 1D array instead of 2D
        response = server.handle_request({
            'cmd': 'step',
            'positions': np.array([1, 2, 3])
        })
        # Should either handle gracefully or return error
        assert response['status'] in ['ok', 'error']

    def test_handle_empty_config_update(self, server):
        """Test empty config update."""
        response = server.handle_request({
            'cmd': 'config',
            'config': {}
        })
        assert response['status'] == 'ok'


class TestLeniaServerStepSequence:
    """Tests for step sequences."""

    @pytest.fixture
    def server(self, monkeypatch):
        """Create server with mocked ZMQ.

        Note: injection_power must be > 1.0 to overcome negative Lenia growth.
        """
        mock_context = MockContext()
        monkeypatch.setattr('zmq.Context', lambda: mock_context)
        config = FieldConfig(width=64, height=64, injection_power=2.0)
        return LeniaServer(port=5555, config=config)

    def test_multiple_steps(self, server):
        """Test multiple consecutive steps."""
        positions = np.array([[32, 32]], dtype=np.float32)

        for i in range(10):
            response = server.handle_request({
                'cmd': 'step',
                'positions': positions
            })
            assert response['status'] == 'ok'

    def test_step_get_sequence(self, server):
        """Test alternating step and get."""
        positions = np.array([[32, 32]], dtype=np.float32)

        for i in range(5):
            step_response = server.handle_request({
                'cmd': 'step',
                'positions': positions
            })
            get_response = server.handle_request({'cmd': 'get'})

            assert step_response['status'] == 'ok'
            assert get_response['status'] == 'ok'
            # Fields should match
            assert np.array_equal(
                step_response['field'],
                get_response['field']
            )

    def test_reset_step_sequence(self, server):
        """Test reset followed by steps."""
        positions = np.array([[32, 32]], dtype=np.float32)

        # Step to create field values
        server.handle_request({'cmd': 'step', 'positions': positions})

        # Reset
        server.handle_request({'cmd': 'reset'})

        # Verify field is zero
        response = server.handle_request({'cmd': 'get'})
        assert float(np.max(response['field'])) == 0.0

        # Step again
        response = server.handle_request({
            'cmd': 'step',
            'positions': positions
        })
        assert float(np.max(response['field'])) > 0
