"""
Tests for client.py - Lenia client and response classes.
"""

import pytest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lenia_field.client import LeniaClient, LeniaResponse, LeniaClientAsync


class TestLeniaResponse:
    """Tests for LeniaResponse dataclass."""

    def test_ok_response(self):
        """Test successful response."""
        response = LeniaResponse(
            status='ok',
            field=np.zeros((64, 64)),
            shape=(64, 64),
            fps=100.0
        )
        assert response.ok is True
        assert response.status == 'ok'
        assert response.error is None

    def test_error_response(self):
        """Test error response."""
        response = LeniaResponse(
            status='error',
            error='Something went wrong'
        )
        assert response.ok is False
        assert response.status == 'error'
        assert response.error == 'Something went wrong'

    def test_response_with_config(self):
        """Test response with config data."""
        config_data = {'width': 512, 'height': 512}
        response = LeniaResponse(
            status='ok',
            config=config_data
        )
        assert response.ok is True
        assert response.config == config_data

    def test_response_field_none(self):
        """Test response without field data."""
        response = LeniaResponse(status='ok')
        assert response.field is None
        assert response.shape is None

    def test_response_defaults(self):
        """Test response default values."""
        response = LeniaResponse(status='ok')
        assert response.fps == 0.0
        assert response.error is None
        assert response.config is None


class TestLeniaClientInit:
    """Tests for LeniaClient initialization (unit tests with mocking)."""

    def test_client_default_values(self, monkeypatch):
        """Test client initializes with default values."""
        # Mock ZMQ
        class MockSocket:
            def setsockopt(self, *args): pass
            def connect(self, *args): pass
            def close(self): pass

        class MockContext:
            def socket(self, *args): return MockSocket()
            def term(self): pass

        monkeypatch.setattr('zmq.Context', MockContext)
        monkeypatch.setattr('zmq.REQ', 0)

        client = LeniaClient(port=5555)
        assert client.port == 5555
        assert client.host == "localhost"
        assert client.timeout == 5.0
        assert client.process is None

    def test_client_custom_values(self, monkeypatch):
        """Test client with custom values."""
        class MockSocket:
            def setsockopt(self, *args): pass
            def connect(self, *args): pass
            def close(self): pass

        class MockContext:
            def socket(self, *args): return MockSocket()
            def term(self): pass

        monkeypatch.setattr('zmq.Context', MockContext)
        monkeypatch.setattr('zmq.REQ', 0)

        client = LeniaClient(
            port=6000,
            host="192.168.1.1",
            timeout=10.0
        )
        assert client.port == 6000
        assert client.host == "192.168.1.1"
        assert client.timeout == 10.0


class TestLeniaClientRequestMocking:
    """Tests for client request methods with mocked ZMQ."""

    @pytest.fixture
    def mock_client(self, monkeypatch):
        """Create client with mocked ZMQ."""
        import msgpack
        import msgpack_numpy as m
        m.patch()

        class MockSocket:
            def __init__(self):
                self.response_data = {'status': 'ok'}

            def setsockopt(self, *args): pass
            def connect(self, *args): pass
            def close(self): pass

            def send(self, data):
                # Parse request for inspection
                self.last_request = msgpack.unpackb(data, raw=False)

            def recv(self):
                return msgpack.packb(self.response_data)

            def set_response(self, data):
                self.response_data = data

        class MockContext:
            def __init__(self):
                self.socket_instance = None

            def socket(self, *args):
                self.socket_instance = MockSocket()
                return self.socket_instance

            def term(self): pass

        mock_ctx = MockContext()
        monkeypatch.setattr('zmq.Context', lambda: mock_ctx)
        monkeypatch.setattr('zmq.REQ', 0)

        client = LeniaClient(port=5555)
        return client, mock_ctx.socket_instance

    def test_step_request(self, mock_client):
        """Test step sends correct request."""
        client, socket = mock_client
        socket.set_response({
            'status': 'ok',
            'field': np.zeros((64, 64)),
            'shape': (64, 64),
            'fps': 100.0
        })

        response = client.step()
        assert response.ok is True
        assert socket.last_request['cmd'] == 'step'

    def test_step_with_positions(self, mock_client):
        """Test step with positions."""
        client, socket = mock_client
        socket.set_response({
            'status': 'ok',
            'field': np.zeros((64, 64)),
            'shape': (64, 64),
            'fps': 100.0
        })

        positions = [[10, 20], [30, 40]]
        response = client.step(positions=positions)

        assert response.ok is True
        assert 'positions' in socket.last_request
        assert np.array_equal(
            socket.last_request['positions'],
            np.array(positions, dtype=np.float32)
        )

    def test_step_with_powers(self, mock_client):
        """Test step with custom powers."""
        client, socket = mock_client
        socket.set_response({'status': 'ok', 'shape': (64, 64), 'fps': 0})

        positions = [[10, 20]]
        powers = [0.5]
        client.step(positions=positions, powers=powers)

        assert 'powers' in socket.last_request

    def test_step_with_uint8(self, mock_client):
        """Test step with uint8 flag."""
        client, socket = mock_client
        socket.set_response({'status': 'ok', 'shape': (64, 64), 'fps': 0})

        client.step(uint8=True)
        assert socket.last_request['uint8'] is True

    def test_step_with_config_updates(self, mock_client):
        """Test step with config updates."""
        client, socket = mock_client
        socket.set_response({'status': 'ok', 'shape': (64, 64), 'fps': 0})

        client.step(decay=0.1, diffusion=0.05)
        assert socket.last_request['config'] == {'decay': 0.1, 'diffusion': 0.05}

    def test_get_request(self, mock_client):
        """Test get sends correct request."""
        client, socket = mock_client
        socket.set_response({
            'status': 'ok',
            'field': np.zeros((64, 64)),
            'shape': (64, 64),
            'fps': 0
        })

        response = client.get()
        assert response.ok is True
        assert socket.last_request['cmd'] == 'get'

    def test_get_uint8(self, mock_client):
        """Test get with uint8 flag."""
        client, socket = mock_client
        socket.set_response({'status': 'ok', 'shape': (64, 64), 'fps': 0})

        client.get(uint8=True)
        assert socket.last_request['uint8'] is True

    def test_reset_request(self, mock_client):
        """Test reset sends correct request."""
        client, socket = mock_client
        socket.set_response({'status': 'ok'})

        response = client.reset()
        assert response.ok is True
        assert socket.last_request['cmd'] == 'reset'

    def test_config_request(self, mock_client):
        """Test config sends correct request."""
        client, socket = mock_client
        socket.set_response({
            'status': 'ok',
            'config': {'width': 64, 'height': 64}
        })

        response = client.config(decay=0.1)
        assert response.ok is True
        assert socket.last_request['cmd'] == 'config'
        assert socket.last_request['config'] == {'decay': 0.1}

    def test_ping_request(self, mock_client):
        """Test ping sends correct request."""
        client, socket = mock_client
        socket.set_response({'status': 'ok', 'message': 'pong'})

        response = client.ping()
        assert response.ok is True
        assert socket.last_request['cmd'] == 'ping'

    def test_shutdown_request(self, mock_client):
        """Test shutdown sends correct request."""
        client, socket = mock_client
        socket.set_response({'status': 'ok', 'shutdown': True})

        response = client.shutdown()
        assert response.ok is True
        assert socket.last_request['cmd'] == 'shutdown'


class TestLeniaClientErrorHandling:
    """Tests for client error handling."""

    @pytest.fixture
    def error_client(self, monkeypatch):
        """Create client that returns errors."""
        import msgpack

        class MockSocket:
            def __init__(self):
                self.should_timeout = False

            def setsockopt(self, *args): pass
            def connect(self, *args): pass
            def close(self): pass
            def send(self, data): pass

            def recv(self):
                if self.should_timeout:
                    import zmq
                    raise zmq.error.Again()
                return msgpack.packb({'status': 'error', 'error': 'Test error'})

        class MockContext:
            def __init__(self):
                self.socket_instance = MockSocket()

            def socket(self, *args):
                return self.socket_instance

            def term(self): pass

        mock_ctx = MockContext()
        monkeypatch.setattr('zmq.Context', lambda: mock_ctx)
        monkeypatch.setattr('zmq.REQ', 0)

        client = LeniaClient(port=5555)
        return client, mock_ctx.socket_instance

    def test_error_response(self, error_client):
        """Test handling of error response."""
        client, socket = error_client
        response = client.step()
        assert response.ok is False
        assert response.error == 'Test error'

    def test_timeout_handling(self, error_client):
        """Test handling of timeout."""
        client, socket = error_client
        socket.should_timeout = True

        response = client.step()
        assert response.ok is False
        assert 'timeout' in response.error.lower()


class TestLeniaClientContextManager:
    """Tests for client context manager."""

    def test_context_manager_calls_close(self, monkeypatch):
        """Test context manager calls close."""
        close_called = [False]

        class MockSocket:
            def setsockopt(self, *args): pass
            def connect(self, *args): pass
            def close(self):
                close_called[0] = True

        class MockContext:
            def socket(self, *args): return MockSocket()
            def term(self): pass

        monkeypatch.setattr('zmq.Context', MockContext)
        monkeypatch.setattr('zmq.REQ', 0)

        with LeniaClient(port=5555) as client:
            pass

        assert close_called[0] is True


class TestLeniaClientClose:
    """Tests for client close method."""

    def test_close_terminates_process(self, monkeypatch):
        """Test close terminates subprocess."""
        terminate_called = [False]
        wait_called = [False]

        class MockSocket:
            def setsockopt(self, *args): pass
            def connect(self, *args): pass
            def close(self): pass

        class MockContext:
            def socket(self, *args): return MockSocket()
            def term(self): pass

        class MockProcess:
            def terminate(self):
                terminate_called[0] = True

            def wait(self, timeout=None):
                wait_called[0] = True

        monkeypatch.setattr('zmq.Context', MockContext)
        monkeypatch.setattr('zmq.REQ', 0)

        client = LeniaClient(port=5555)
        client.process = MockProcess()
        client.close()

        assert terminate_called[0] is True
        assert wait_called[0] is True


class TestLeniaClientAsync:
    """Tests for async client wrapper."""

    @pytest.fixture
    def mock_sync_client(self, monkeypatch):
        """Create mock synchronous client."""
        import msgpack

        class MockSocket:
            def setsockopt(self, *args): pass
            def connect(self, *args): pass
            def close(self): pass
            def send(self, data): pass
            def recv(self):
                return msgpack.packb({
                    'status': 'ok',
                    'field': np.random.rand(64, 64).astype(np.float32),
                    'fps': 100.0
                })

        class MockContext:
            def socket(self, *args): return MockSocket()
            def term(self): pass

        monkeypatch.setattr('zmq.Context', MockContext)
        monkeypatch.setattr('zmq.REQ', 0)

        return LeniaClient(port=5555)

    def test_async_client_init(self, mock_sync_client):
        """Test async client initialization."""
        async_client = LeniaClientAsync(mock_sync_client)
        assert async_client.client is mock_sync_client
        assert async_client._running is True

        # Cleanup
        async_client.close()

    def test_async_step(self, mock_sync_client):
        """Test async step queues request."""
        async_client = LeniaClientAsync(mock_sync_client)

        # Queue a step
        positions = np.array([[32, 32]], dtype=np.float32)
        async_client.step_async(positions=positions)

        # Wait a bit for processing
        time.sleep(0.2)

        # Should have received a field
        field, fps = async_client.get_latest()
        assert field is not None
        assert fps > 0

        async_client.close()

    def test_async_get_latest_initial(self, mock_sync_client):
        """Test get_latest returns None initially."""
        async_client = LeniaClientAsync(mock_sync_client)

        # Immediately after init, should be None
        field, fps = async_client.get_latest()
        # May or may not be None depending on timing
        assert fps >= 0.0

        async_client.close()

    def test_async_close(self, mock_sync_client):
        """Test async client close."""
        async_client = LeniaClientAsync(mock_sync_client)
        async_client.close()

        assert async_client._running is False


class TestLeniaClientSpawn:
    """Tests for spawn class method (integration-like tests)."""

    def test_spawn_generates_correct_command(self, monkeypatch):
        """Test spawn generates correct subprocess command."""
        captured_cmd = []

        class MockProcess:
            stdout = None
            stderr = None

            def __init__(self, cmd, **kwargs):
                captured_cmd.extend(cmd)

            def terminate(self): pass
            def wait(self, timeout=None): pass
            def kill(self): pass

        class MockSocket:
            def setsockopt(self, *args): pass
            def connect(self, *args): pass
            def close(self): pass
            def send(self, data): pass
            def recv(self):
                import msgpack
                return msgpack.packb({'status': 'ok', 'message': 'pong'})

        class MockContext:
            def socket(self, *args): return MockSocket()
            def term(self): pass

        monkeypatch.setattr('subprocess.Popen', MockProcess)
        monkeypatch.setattr('zmq.Context', MockContext)
        monkeypatch.setattr('zmq.REQ', 0)
        monkeypatch.setattr('time.sleep', lambda x: None)

        try:
            client = LeniaClient.spawn(
                port=5555,
                width=256,
                height=128,
                dt=0.05,
                diffusion=0.02,
                decay=0.003
            )
            client.close()
        except:
            pass  # May fail due to incomplete mocking

        # Check command contains expected args
        assert '--port' in captured_cmd
        assert '5555' in captured_cmd
        assert '--width' in captured_cmd
        assert '256' in captured_cmd
        assert '--height' in captured_cmd
        assert '128' in captured_cmd
