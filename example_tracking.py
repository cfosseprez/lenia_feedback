"""
Example: Integration with a tracking pipeline.

This shows how to couple Lenia field with real-time organism tracking.
Each tracked organism "excretes" morphogen at its position.

For your Rosetta/SwarmTracker setup, you would:
1. Run your image analysis pipeline
2. Extract organism positions
3. Feed them to the Lenia field
4. Project the field back to the organisms
"""

import numpy as np
import time
from typing import List, Tuple
from dataclasses import dataclass

from lenia_field import LeniaClient, FieldConfig


@dataclass
class TrackedOrganism:
    """Simulated tracked organism."""
    id: int
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy]
    excretion_rate: float = 1.0


class MockTracker:
    """
    Mock tracker that simulates organism movement.
    Replace this with your actual SwarmTracker output.
    """
    
    def __init__(self, n_organisms: int, field_size: Tuple[int, int]):
        self.field_size = field_size
        self.organisms: List[TrackedOrganism] = []
        
        for i in range(n_organisms):
            pos = np.random.rand(2) * np.array(field_size)
            vel = (np.random.rand(2) - 0.5) * 2
            self.organisms.append(TrackedOrganism(
                id=i,
                position=pos,
                velocity=vel,
                excretion_rate=np.random.uniform(0.5, 1.5)
            ))
    
    def update(self) -> List[TrackedOrganism]:
        """Update organism positions (random walk + boundary reflection)."""
        for org in self.organisms:
            # Random walk
            org.velocity += (np.random.rand(2) - 0.5) * 0.5
            org.velocity = np.clip(org.velocity, -3, 3)
            org.position += org.velocity
            
            # Boundary reflection
            for i in range(2):
                if org.position[i] < 0:
                    org.position[i] = -org.position[i]
                    org.velocity[i] *= -1
                elif org.position[i] >= self.field_size[i]:
                    org.position[i] = 2 * self.field_size[i] - org.position[i] - 1
                    org.velocity[i] *= -1
        
        return self.organisms


class LeniaFieldIntegration:
    """
    Integrates Lenia field with tracking data.
    
    This is the main class you'd use in your pipeline:
    
        integration = LeniaFieldIntegration(width=512, height=512)
        
        while running:
            # Your tracking pipeline
            positions = tracker.get_positions()
            
            # Update Lenia and get projection
            field = integration.update(positions)
            
            # Project field to organisms
            projector.send(field)
    """
    
    def __init__(self, 
                 width: int = 512,
                 height: int = 512,
                 port: int = 5558,
                 **lenia_params):
        """
        Initialize Lenia field integration.
        
        Args:
            width: Field width (should match your projection resolution)
            height: Field height
            port: ZMQ port for server communication
            **lenia_params: Additional parameters for Lenia
        """
        self.width = width
        self.height = height
        
        # Default parameters tuned for protist-scale dynamics
        default_params = {
            'dt': 0.1,
            'diffusion': 0.008,
            'decay': 0.003,
        }
        default_params.update(lenia_params)
        
        # Spawn server
        self.client = LeniaClient.spawn(
            port=port,
            width=width,
            height=height,
            **default_params
        )
        
        # Configure injection parameters
        self.client.config(
            injection_radius=6.0,
            injection_power=0.1,
            growth_mu=0.14,
            growth_sigma=0.015
        )
        
        self._last_field: np.ndarray = np.zeros((height, width))
        
    def update(self, 
               positions: np.ndarray,
               powers: np.ndarray = None) -> np.ndarray:
        """
        Update Lenia field with new organism positions.
        
        Args:
            positions: (N, 2) array of [x, y] positions
            powers: (N,) array of excretion powers (optional)
            
        Returns:
            Current field as (height, width) float32 array in [0, 1]
        """
        if len(positions) == 0:
            response = self.client.step()
        else:
            response = self.client.step(positions=positions, powers=powers)
        
        if response.ok and response.field is not None:
            self._last_field = response.field
        
        return self._last_field
    
    def get_field_uint8(self) -> np.ndarray:
        """Get field as uint8 for projection."""
        return (self._last_field * 255).astype(np.uint8)
    
    def sample_at_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Sample field values at given positions.
        Useful for measuring morphogen concentration at organism locations.
        
        Args:
            positions: (N, 2) array of [x, y] positions
            
        Returns:
            (N,) array of field values at those positions
        """
        positions = np.asarray(positions)
        x = np.clip(positions[:, 0].astype(int), 0, self.width - 1)
        y = np.clip(positions[:, 1].astype(int), 0, self.height - 1)
        return self._last_field[y, x]
    
    def get_gradient_at_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Get field gradient at positions.
        Useful for chemotaxis-like behavior analysis.
        
        Args:
            positions: (N, 2) array of [x, y] positions
            
        Returns:
            (N, 2) array of gradient vectors [dx, dy]
        """
        # Compute gradient
        gy, gx = np.gradient(self._last_field)
        
        positions = np.asarray(positions)
        x = np.clip(positions[:, 0].astype(int), 0, self.width - 1)
        y = np.clip(positions[:, 1].astype(int), 0, self.height - 1)
        
        return np.stack([gx[y, x], gy[y, x]], axis=1)
    
    def reset(self):
        """Reset field to zero."""
        self.client.reset()
        self._last_field = np.zeros((self.height, self.width))
    
    def set_params(self, **kwargs):
        """Update Lenia parameters."""
        self.client.config(**kwargs)
    
    def close(self):
        """Cleanup."""
        self.client.shutdown()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def main():
    """Demo with mock tracker."""
    
    width, height = 256, 256
    n_organisms = 10
    
    # Create mock tracker
    tracker = MockTracker(n_organisms, (width, height))
    
    # Create integration
    print("Starting Lenia field integration...")
    integration = LeniaFieldIntegration(
        width=width,
        height=height,
        port=5559
    )
    
    try:
        # Optional: visualization
        try:
            import cv2
            has_viz = True
            cv2.namedWindow('Lenia + Tracking', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Lenia + Tracking', 512, 512)
        except ImportError:
            has_viz = False
            print("OpenCV not available, running without visualization")
        
        print(f"Running with {n_organisms} simulated organisms...")
        print("Press 'q' to quit (if visualization enabled)")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Update tracker (replace with your actual tracking)
            organisms = tracker.update()
            
            # Extract positions and excretion rates
            positions = np.array([org.position for org in organisms])
            powers = np.array([org.excretion_rate for org in organisms]) * 0.1
            
            # Update Lenia field
            field = integration.update(positions, powers)
            
            # Sample field at organism positions (for analysis)
            concentrations = integration.sample_at_positions(positions)
            
            # Print stats periodically
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Frame {frame_count}: {fps:.1f} FPS, "
                      f"field_max={field.max():.3f}, "
                      f"mean_concentration={concentrations.mean():.3f}")
            
            # Visualization
            if has_viz:
                # Apply colormap
                field_uint8 = (field * 255).astype(np.uint8)
                frame = cv2.applyColorMap(field_uint8, cv2.COLORMAP_INFERNO)
                
                # Draw organism positions
                for org in organisms:
                    x, y = int(org.position[0]), int(org.position[1])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
                cv2.imshow('Lenia + Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Run for fixed number of frames without viz
                if frame_count >= 500:
                    break
                time.sleep(0.01)
    
    finally:
        if has_viz:
            cv2.destroyAllWindows()
        integration.close()
        print("Done!")


if __name__ == "__main__":
    main()
