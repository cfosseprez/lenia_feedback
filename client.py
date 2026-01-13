"""
Client for interacting with Lenia server subprocess.
Non-blocking interface for real-time applications.
"""

import zmq
import numpy as np
import msgpack
import msgpack_numpy as m
import subprocess
import time
import sys
import os
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass

# Patch msgpack for numpy support
m.patch()


@dataclass
class LeniaResponse:
    """Response from Lenia server."""
    status: str
    field: Optional[np.ndarray] = None
    shape: Optional[Tuple[int, int]] = None
    fps: float = 0.0
    error: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    
    @property
    def ok(self) -> bool:
        return self.status == 'ok'


class LeniaClient:
    """
    Client for communicating with Lenia server subprocess.
    
    Usage:
        # Start server and connect
        client = LeniaClient.spawn(width=512, height=512)
        
        # Or connect to existing server
        client = LeniaClient(port=5555)
        
        # Step simulation with injection points
        response = client.step(positions=[[100, 100], [200, 200]])
        field = response.field
        
        # Update parameters on the fly
        client.config(diffusion=0.02, decay=0.005)
        
        # Cleanup
        client.shutdown()
    """
    
    def __init__(self, 
                 port: int = 5555,
                 host: str = "localhost",
                 timeout: float = 5.0):
        """
        Connect to existing Lenia server.
        
        Args:
            port: ZMQ port
            host: Server hostname
            timeout: Request timeout in seconds
        """
        self.port = port
        self.host = host
        self.timeout = timeout
        self.process: Optional[subprocess.Popen] = None
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout * 1000))
        self.socket.connect(f"tcp://{host}:{port}")
        
    @classmethod
    def spawn(cls,
              port: int = 5555,
              width: int = 512,
              height: int = 512,
              dt: float = 0.1,
              diffusion: float = 0.01,
              decay: float = 0.001,
              startup_timeout: float = 10.0,
              **kwargs) -> 'LeniaClient':
        """
        Spawn a new Lenia server subprocess and connect to it.
        
        Args:
            port: ZMQ port for communication
            width: Field width
            height: Field height
            dt: Time step
            diffusion: Diffusion rate
            decay: Decay rate
            startup_timeout: Max time to wait for server startup
            **kwargs: Additional arguments passed to LeniaClient
            
        Returns:
            Connected LeniaClient instance
        """
        # Find server script
        server_script = os.path.join(os.path.dirname(__file__), 'server.py')
        
        # Start server process
        cmd = [
            sys.executable, server_script,
            '--port', str(port),
            '--width', str(width),
            '--height', str(height),
            '--dt', str(dt),
            '--diffusion', str(diffusion),
            '--decay', str(decay)
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(0.5)  # Give it a moment to initialize
        
        # Create client
        client = cls(port=port, **kwargs)
        client.process = process
        
        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < startup_timeout:
            try:
                response = client.ping()
                if response.ok:
                    return client
            except Exception:
                pass
            time.sleep(0.1)
        
        # Failed to connect
        process.terminate()
        raise RuntimeError(f"Failed to connect to Lenia server within {startup_timeout}s")
    
    def _send_request(self, request: Dict[str, Any]) -> LeniaResponse:
        """Send request and receive response."""
        try:
            self.socket.send(msgpack.packb(request))
            message = self.socket.recv()
            data = msgpack.unpackb(message, raw=False)
            
            return LeniaResponse(
                status=data.get('status', 'error'),
                field=data.get('field'),
                shape=tuple(data['shape']) if 'shape' in data else None,
                fps=data.get('fps', 0.0),
                error=data.get('error'),
                config=data.get('config')
            )
        except zmq.error.Again:
            return LeniaResponse(status='error', error='Request timeout')
        except Exception as e:
            return LeniaResponse(status='error', error=str(e))
    
    def step(self,
             positions: Optional[Union[np.ndarray, list]] = None,
             powers: Optional[Union[np.ndarray, list]] = None,
             uint8: bool = False,
             return_field: bool = True,
             **config_updates) -> LeniaResponse:
        """
        Advance simulation by one step.
        
        Args:
            positions: (N, 2) array of (x, y) injection positions
            powers: (N,) array of power per position (default: use config)
            uint8: Return field as uint8 (0-255) instead of float
            return_field: Whether to return field data
            **config_updates: Config parameters to update for this step
            
        Returns:
            LeniaResponse with field data
        """
        request = {
            'cmd': 'step',
            'uint8': uint8,
            'return_field': return_field
        }
        
        if positions is not None:
            request['positions'] = np.asarray(positions, dtype=np.float32)
        if powers is not None:
            request['powers'] = np.asarray(powers, dtype=np.float32)
        if config_updates:
            request['config'] = config_updates
            
        return self._send_request(request)
    
    def get(self, uint8: bool = False) -> LeniaResponse:
        """
        Get current field without stepping simulation.
        
        Args:
            uint8: Return as uint8 instead of float
            
        Returns:
            LeniaResponse with field data
        """
        return self._send_request({
            'cmd': 'get',
            'uint8': uint8
        })
    
    def reset(self) -> LeniaResponse:
        """Reset field to zero."""
        return self._send_request({'cmd': 'reset'})
    
    def config(self, **kwargs) -> LeniaResponse:
        """
        Update configuration parameters.
        
        Available parameters:
            dt, diffusion, decay, injection_radius, injection_power,
            growth_mu, growth_sigma, growth_amplitude
            
        Returns:
            LeniaResponse with current config
        """
        return self._send_request({
            'cmd': 'config',
            'config': kwargs
        })
    
    def ping(self) -> LeniaResponse:
        """Health check."""
        return self._send_request({'cmd': 'ping'})
    
    def shutdown(self) -> LeniaResponse:
        """Shutdown server."""
        response = self._send_request({'cmd': 'shutdown'})
        self.close()
        return response
    
    def close(self):
        """Close connection and cleanup."""
        self.socket.close()
        self.context.term()
        
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class LeniaClientAsync:
    """
    Async client with non-blocking step() using a background thread.
    Useful when you need fire-and-forget injection.
    """
    
    def __init__(self, client: LeniaClient):
        self.client = client
        self._latest_field: Optional[np.ndarray] = None
        self._latest_fps: float = 0.0
        
        import threading
        import queue
        
        self._request_queue: queue.Queue = queue.Queue()
        self._running = True
        
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        
    def _worker(self):
        """Background worker that processes requests."""
        import queue
        
        while self._running:
            try:
                request = self._request_queue.get(timeout=0.1)
                response = self.client._send_request(request)
                if response.ok and response.field is not None:
                    self._latest_field = response.field
                    self._latest_fps = response.fps
            except queue.Empty:
                continue
            except Exception:
                pass
    
    def step_async(self, 
                   positions: Optional[np.ndarray] = None,
                   powers: Optional[np.ndarray] = None,
                   uint8: bool = False):
        """Queue a step request (non-blocking)."""
        request = {'cmd': 'step', 'uint8': uint8, 'return_field': True}
        if positions is not None:
            request['positions'] = np.asarray(positions, dtype=np.float32)
        if powers is not None:
            request['powers'] = np.asarray(powers, dtype=np.float32)
        self._request_queue.put(request)
    
    def get_latest(self) -> Tuple[Optional[np.ndarray], float]:
        """Get latest field and FPS (non-blocking)."""
        return self._latest_field, self._latest_fps
    
    def close(self):
        """Stop worker and close client."""
        self._running = False
        self._thread.join(timeout=1.0)
        self.client.close()


# Quick test
if __name__ == "__main__":
    print("Spawning Lenia server...")
    
    with LeniaClient.spawn(
        width=256,
        height=256,
        port=5556
    ) as client:
        print(f"Connected! Ping: {client.ping().ok}")
        
        # Run a few steps
        for i in range(10):
            positions = np.random.rand(5, 2) * 256
            response = client.step(positions=positions)
            print(f"Step {i}: fps={response.fps:.1f}, field_max={response.field.max():.3f}")
        
        # Update config
        result = client.config(injection_power=0.2, decay=0.01)
        print(f"Config: {result.config}")
        
        # More steps
        for i in range(10):
            positions = np.random.rand(3, 2) * 256
            response = client.step(positions=positions)
            print(f"Step {i}: fps={response.fps:.1f}, field_max={response.field.max():.3f}")
        
    print("Done!")
