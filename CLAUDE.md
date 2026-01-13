# CLAUDE.md - lenia_field Development Guide

## Overview

Real-time Lenia-inspired morphogen field with external stimulus injection. Designed for bio-hybrid systems where tracked organisms "excrete" morphogen into a reactive medium. The system receives real-time position data from tracking pipelines and integrates them into a JAX-accelerated Lenia simulation.

## Architecture

```
┌─────────────────┐     ZMQ/msgpack    ┌─────────────────┐
│  Main Process   │◄──────────────────►│  Lenia Server   │
│  (Tracker)      │                    │  (JAX/GPU)      │
└─────────────────┘                    └─────────────────┘
        │                                      │
        ▼                                      ▼
   positions[]  ─────────────────────►   field[H, W]
```

**Key Flow**: Tracking positions → ZMQ injection → Lenia step → Field response

## File Structure

- `core.py` - JAX-based Lenia field simulation (LeniaField, FieldConfig)
- `server.py` - ZMQ server wrapping LeniaField for non-blocking operation
- `client.py` - Client API for spawning/connecting to server
- `__init__.py` - Package exports

## Real-Time Position Integration

The core integration point is the `step()` method which accepts position arrays:

```python
# From tracking pipeline
positions = np.array([[x1, y1], [x2, y2], ...])  # (N, 2) float32
powers = np.array([p1, p2, ...])  # (N,) optional injection strength

# Inject into field
response = client.step(positions=positions, powers=powers)
field = response.field  # (H, W) float32 in [0, 1]
```

## Key Components

### FieldConfig (core.py:16)
Dataclass with all simulation parameters:
- Grid: `width`, `height`
- Lenia kernel: `kernel_radius`, `kernel_sigma`
- Growth function: `growth_mu`, `growth_sigma`, `growth_amplitude`
- Dynamics: `dt`, `diffusion`, `decay`
- Injection: `injection_radius`, `injection_power`

### LeniaField (core.py:148)
Direct simulation class (blocking):
- `step(positions, powers)` - Advance one step with optional injection
- `reset()` - Clear field to zero
- `get_field()` / `get_field_uint8()` - Read current field
- `update_config(**kwargs)` - Modify parameters at runtime

### LeniaClient (client.py:36)
Non-blocking client via subprocess:
- `spawn(width, height, ...)` - Start server subprocess
- `step(positions, powers, ...)` - Send step command
- `config(**kwargs)` - Update parameters
- `reset()` / `shutdown()` - Control server lifecycle

### ZMQ Protocol
Commands: `step`, `get`, `reset`, `config`, `ping`, `shutdown`
Serialization: msgpack with numpy support

## Build Commands

```bash
# Install in development mode
pip install -e .

# Install with visualization support
pip install -e ".[viz]"

# Install with dev tools (pytest, etc.)
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core.py -v
```

## Known Issues

1. **Package structure**: Files are in root but pyproject.toml expects `lenia_field/` subdirectory. Move files or update config for proper installation.

2. **Import in server.py**: Line 18 uses `from core import ...` (absolute) but should be `from .core import ...` (relative) for package installation.

## Fixed Bugs

1. **JAX JIT static_argnums** (core.py:114): `create_injection_mask` was missing `shape` in static_argnums. Fixed by changing `@partial(jit, static_argnums=(3,))` to `@partial(jit, static_argnums=(2, 3))`.

2. **JAX convolve2d boundary** (core.py:197): `jax.scipy.signal.convolve2d` doesn't support `boundary='wrap'`. Fixed by using FFT convolution for diffusion term.

## Important: Lenia Growth Dynamics

The Lenia growth function returns -1 when the potential (convolved field) is 0. This means:
- When the field is empty, the growth term actively pushes values negative
- `injection_power` must be > 1.0 to overcome this negative growth and produce positive field values
- Default `injection_power=0.1` will NOT produce visible injection on an empty field

Recommended for testing: `injection_power=2.0` or higher.

## Testing Notes

- Core tests can run without ZMQ (test `LeniaField` directly)
- Server/client tests require ZMQ and subprocess spawning
- Use pytest fixtures for shared setup (field configs, mock positions)
- Integration tests should verify full pipeline: spawn → inject → read → shutdown

## Performance Tuning

- FFT convolution (`use_fft=True`) faster for kernel_radius > 7
- JIT compilation warms up on first call
- Target: ~200+ FPS on GPU at 512x512
