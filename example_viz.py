"""
Example: Real-time visualization of Lenia field with mouse-driven injection.

Run with:
    python example_viz.py
    
Click and drag to inject morphogen.
Press 'r' to reset, 'q' to quit.
Press 1-9 to adjust injection power.
"""

import numpy as np
import cv2
import time
from lenia_field import LeniaClient, FieldConfig


def apply_colormap(field: np.ndarray) -> np.ndarray:
    """Apply a nice colormap to the field."""
    # Normalize and convert to uint8
    field_norm = np.clip(field, 0, 1)
    field_uint8 = (field_norm * 255).astype(np.uint8)
    
    # Apply colormap (INFERNO looks nice for this)
    colored = cv2.applyColorMap(field_uint8, cv2.COLORMAP_INFERNO)
    
    return colored


def main():
    # Configuration
    width, height = 512, 512
    display_scale = 1  # Scale for display window
    
    # Spawn server
    print("Starting Lenia server...")
    client = LeniaClient.spawn(
        port=5557,
        width=width,
        height=height,
        dt=0.1,
        diffusion=0.005,
        decay=0.002
    )
    
    # Configure initial parameters
    client.config(
        injection_radius=8.0,
        injection_power=0.15,
        growth_mu=0.15,
        growth_sigma=0.017
    )
    
    # Mouse state
    mouse_positions = []
    injection_power = 0.15
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_positions
        if event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            # Scale back if display is scaled
            x = x // display_scale
            y = y // display_scale
            mouse_positions.append([x, y])
        elif event == cv2.EVENT_LBUTTONDOWN:
            x = x // display_scale
            y = y // display_scale
            mouse_positions.append([x, y])
    
    # Create window
    cv2.namedWindow('Lenia Field', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Lenia Field', mouse_callback)
    cv2.resizeWindow('Lenia Field', width * display_scale, height * display_scale)
    
    print("\nControls:")
    print("  Click/drag: Inject morphogen")
    print("  1-9: Adjust injection power")
    print("  r: Reset field")
    print("  d/D: Decrease/Increase diffusion")
    print("  c/C: Decrease/Increase decay")
    print("  q: Quit")
    print()
    
    fps_display = 0.0
    frame_times = []
    
    try:
        while True:
            frame_start = time.time()
            
            # Collect positions and step
            if mouse_positions:
                positions = np.array(mouse_positions)
                powers = np.ones(len(positions)) * injection_power
                response = client.step(positions=positions, powers=powers)
                mouse_positions = []
            else:
                response = client.step()
            
            if not response.ok:
                print(f"Error: {response.error}")
                continue
            
            # Get field and apply colormap
            field = response.field
            frame = apply_colormap(field)
            
            # Scale for display
            if display_scale > 1:
                frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale,
                                   interpolation=cv2.INTER_NEAREST)
            
            # Add FPS overlay
            fps_text = f"Server: {response.fps:.0f} FPS | Power: {injection_power:.2f}"
            cv2.putText(frame, fps_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show
            cv2.imshow('Lenia Field', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                client.reset()
                print("Field reset")
            elif ord('1') <= key <= ord('9'):
                injection_power = (key - ord('0')) * 0.05
                client.config(injection_power=injection_power)
                print(f"Injection power: {injection_power:.2f}")
            elif key == ord('d'):
                result = client.config()
                new_diff = max(0.001, result.config['diffusion'] - 0.002)
                client.config(diffusion=new_diff)
                print(f"Diffusion: {new_diff:.4f}")
            elif key == ord('D'):
                result = client.config()
                new_diff = min(0.1, result.config['diffusion'] + 0.002)
                client.config(diffusion=new_diff)
                print(f"Diffusion: {new_diff:.4f}")
            elif key == ord('c'):
                result = client.config()
                new_decay = max(0.0001, result.config['decay'] - 0.001)
                client.config(decay=new_decay)
                print(f"Decay: {new_decay:.4f}")
            elif key == ord('C'):
                result = client.config()
                new_decay = min(0.1, result.config['decay'] + 0.001)
                client.config(decay=new_decay)
                print(f"Decay: {new_decay:.4f}")
            
            # Track frame time
            frame_times.append(time.time() - frame_start)
            if len(frame_times) > 30:
                frame_times.pop(0)
                fps_display = 1.0 / np.mean(frame_times)
    
    finally:
        cv2.destroyAllWindows()
        client.shutdown()
        print("Done!")


if __name__ == "__main__":
    main()
