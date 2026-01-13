# lenia_field

Real-time Lenia-inspired morphogen field with external stimulus injection. Designed for bio-hybrid systems where tracked organisms "excrete" morphogen into a reactive medium.

## Features

- **JAX-accelerated** - GPU support via JIT compilation
- **Non-blocking** - Runs as subprocess with ZMQ communication
- **Real-time** - ~200+ FPS on GPU at 512x512 resolution
- **Tunable dynamics** - Lenia growth function + diffusion + decay
- **External injection** - Feed positions from tracking pipeline

## Installation

```bash
# Clone or copy the package
cd lenia_field

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .

# For GPU support (CUDA 12)
pip install jax[cuda12]
```

## Quick Start

### As subprocess (recommended for real-time)

```python
from lenia_field import LeniaClient
import numpy as np

# Spawn server subprocess
client = LeniaClient.spawn(width=512, height=512)

# Inject morphogen at positions
positions = np.array([[100, 100], [200, 200], [300, 150]])
response = client.step(positions=positions)

# Get field for projection
field = response.field  # (512, 512) float32 in [0, 1]
field_uint8 = (field * 255).astype(np.uint8)  # For projection

# Update parameters on the fly
client.config(
    diffusion=0.02,
    decay=0.005,
    injection_power=0.2
)

# Cleanup
client.shutdown()
```

### Direct usage (blocking)

```python
from lenia_field import LeniaField, FieldConfig
import numpy as np

config = FieldConfig(
    width=512,
    height=512,
    dt=0.1,
    diffusion=0.01,
    decay=0.002
)

field = LeniaField(config)
result = field.step(positions=np.array([[100, 100]]))
```

## Integration with Tracking Pipeline

```python
from lenia_field import LeniaClient

class MyPipeline:
    def __init__(self):
        self.lenia = LeniaClient.spawn(width=512, height=512)
        self.tracker = MySwarmTracker()  # Your tracker
        self.projector = MyProjector()    # Your projector
        
    def run_frame(self, image):
        # Track organisms
        detections = self.tracker.process(image)
        positions = np.array([[d.x, d.y] for d in detections])
        
        # Update Lenia field
        response = self.lenia.step(positions=positions)
        
        # Project field
        self.projector.send(response.field)
        
        return response.field
```

## Parameters

### FieldConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width`, `height` | 512 | Field dimensions |
| `kernel_radius` | 13 | Lenia kernel radius |
| `kernel_sigma` | 0.5 | Kernel shape (ring width) |
| `growth_mu` | 0.15 | Growth function center |
| `growth_sigma` | 0.015 | Growth function width |
| `dt` | 0.1 | Time step |
| `diffusion` | 0.01 | Extra diffusion rate |
| `decay` | 0.001 | Passive decay rate |
| `injection_radius` | 5.0 | Injection spot radius |
| `injection_power` | 0.1 | Default injection intensity |

### Tuning Tips

- **Fast spreading**: Increase `diffusion`, increase `kernel_radius`
- **Persistent patterns**: Decrease `decay`, tune `growth_mu` and `growth_sigma`
- **Sharp spots**: Decrease `injection_radius`, increase `injection_power`
- **Lenia-like patterns**: Set `diffusion=0`, tune growth function

## Protocol

The server uses ZMQ REQ/REP with msgpack serialization.

Commands:
- `step` - Advance simulation, inject at positions
- `get` - Return field without stepping
- `reset` - Reset field to zero
- `config` - Update parameters
- `ping` - Health check
- `shutdown` - Stop server

## Examples

```bash
# Interactive visualization
python example_viz.py

# Tracking integration demo
python example_tracking.py
```

## Architecture

```
┌─────────────────┐     ZMQ      ┌─────────────────┐
│  Main Process   │◄────────────►│  Lenia Server   │
│  (Your tracker) │   msgpack    │  (JAX/GPU)      │
└─────────────────┘              └─────────────────┘
        │                                │
        ▼                                ▼
   positions[]                    field[H, W]
```

## License

MIT
