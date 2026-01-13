"""
Server process for running Lenia field simulation.
Communicates via ZMQ for non-blocking interaction.
"""

import zmq
import numpy as np
import msgpack
import msgpack_numpy as m
import time
import argparse
from typing import Dict, Any, Optional
import sys

# Patch msgpack for numpy support
m.patch()

from core import LeniaField, FieldConfig


class LeniaServer:
    """
    ZMQ-based server for Lenia field simulation.
    
    Protocol:
        Request (msgpack dict):
            - 'cmd': str - command name
            - 'positions': np.ndarray (N, 2) - optional injection positions
            - 'powers': np.ndarray (N,) - optional injection powers
            - 'config': dict - optional config updates
            
        Response (msgpack dict):
            - 'status': str - 'ok' or 'error'
            - 'field': np.ndarray - current field (if requested)
            - 'field_uint8': np.ndarray - current field as uint8
            - 'shape': tuple - field dimensions
            - 'fps': float - current frame rate
            - 'error': str - error message (if status == 'error')
    
    Commands:
        - 'step': advance simulation, inject at positions, return field
        - 'get': return current field without stepping
        - 'reset': reset field to zero
        - 'config': update configuration parameters
        - 'ping': health check
        - 'shutdown': stop server
    """
    
    def __init__(self, 
                 port: int = 5555,
                 config: Optional[FieldConfig] = None):
        self.port = port
        self.config = config or FieldConfig()
        self.field = LeniaField(self.config)
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
        # Stats
        self.step_count = 0
        self.last_time = time.time()
        self.fps = 0.0
        
        print(f"LeniaServer started on port {port}")
        print(f"Field size: {self.config.width}x{self.config.height}")
        
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming request."""
        cmd = request.get('cmd', 'step')
        
        try:
            if cmd == 'step':
                return self._handle_step(request)
            elif cmd == 'get':
                return self._handle_get(request)
            elif cmd == 'reset':
                return self._handle_reset(request)
            elif cmd == 'config':
                return self._handle_config(request)
            elif cmd == 'ping':
                return {'status': 'ok', 'message': 'pong'}
            elif cmd == 'shutdown':
                return {'status': 'ok', 'shutdown': True}
            else:
                return {'status': 'error', 'error': f'Unknown command: {cmd}'}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _handle_step(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle step command."""
        positions = request.get('positions')
        powers = request.get('powers')
        return_uint8 = request.get('uint8', False)
        return_field = request.get('return_field', True)
        
        # Update config if provided
        if 'config' in request:
            self.field.update_config(**request['config'])
        
        # Step simulation
        field = self.field.step(positions, powers)
        
        # Update stats
        self.step_count += 1
        now = time.time()
        if now - self.last_time > 0.5:
            self.fps = self.step_count / (now - self.last_time)
            self.step_count = 0
            self.last_time = now
        
        response = {
            'status': 'ok',
            'shape': field.shape,
            'fps': self.fps
        }
        
        if return_field:
            if return_uint8:
                response['field'] = self.field.get_field_uint8()
            else:
                response['field'] = field
                
        return response
    
    def _handle_get(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get command (return field without stepping)."""
        return_uint8 = request.get('uint8', False)
        
        if return_uint8:
            field = self.field.get_field_uint8()
        else:
            field = self.field.get_field()
            
        return {
            'status': 'ok',
            'field': field,
            'shape': field.shape,
            'fps': self.fps
        }
    
    def _handle_reset(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reset command."""
        self.field.reset()
        return {'status': 'ok'}
    
    def _handle_config(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle config update command."""
        config_updates = request.get('config', {})
        self.field.update_config(**config_updates)
        
        return {
            'status': 'ok',
            'config': {
                'width': self.field.config.width,
                'height': self.field.config.height,
                'dt': self.field.config.dt,
                'diffusion': self.field.config.diffusion,
                'decay': self.field.config.decay,
                'injection_radius': self.field.config.injection_radius,
                'injection_power': self.field.config.injection_power,
                'growth_mu': self.field.config.growth_mu,
                'growth_sigma': self.field.config.growth_sigma,
            }
        }
    
    def run(self):
        """Main server loop."""
        print("Server ready, waiting for requests...")
        
        while True:
            try:
                # Receive request
                message = self.socket.recv()
                request = msgpack.unpackb(message, raw=False)
                
                # Handle request
                response = self.handle_request(request)
                
                # Check for shutdown
                if response.get('shutdown'):
                    self.socket.send(msgpack.packb(response))
                    break
                
                # Send response
                self.socket.send(msgpack.packb(response))
                
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Error: {e}")
                try:
                    self.socket.send(msgpack.packb({
                        'status': 'error',
                        'error': str(e)
                    }))
                except:
                    pass
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.socket.close()
        self.context.term()
        print("Server stopped")


def main():
    parser = argparse.ArgumentParser(description='Lenia Field Server')
    parser.add_argument('--port', type=int, default=5555, help='ZMQ port')
    parser.add_argument('--width', type=int, default=512, help='Field width')
    parser.add_argument('--height', type=int, default=512, help='Field height')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--diffusion', type=float, default=0.01, help='Diffusion rate')
    parser.add_argument('--decay', type=float, default=0.001, help='Decay rate')
    
    args = parser.parse_args()
    
    config = FieldConfig(
        width=args.width,
        height=args.height,
        dt=args.dt,
        diffusion=args.diffusion,
        decay=args.decay
    )
    
    server = LeniaServer(port=args.port, config=config)
    server.run()


if __name__ == "__main__":
    main()
