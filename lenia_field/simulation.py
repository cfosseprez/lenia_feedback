"""
Simulation module for visual testing of Lenia field with simulated agents.

Provides an interactive demo where agents perform directed random walk,
injecting morphogen into the Lenia field as they move.

Usage:
    # Run with default config (from lenia_field/config/default.yaml)
    python simulation.py

    # Run with custom config file
    python simulation.py --config my_config.yaml

    # Run with runtime overrides (use double underscore for nesting)
    python simulation.py --override field__width=1024 --override agent__speed=3.0

Controls:
    SPACE: Pause/Resume
    r: Reset simulation
    a: Toggle add-agent mode
    1-9: Set injection power
    d/D: Decrease/Increase diffusion
    c/C: Decrease/Increase decay
    s/S: Decrease/Increase agent speed
    m: Toggle metrics logging (debug)
    v: Toggle resource/field view (Suzuki mode)
    h: Print help
    q: Quit
"""

import numpy as np
import time
import csv
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from collections import deque

# Support both direct execution and package import
try:
    from .core import LeniaField, FieldConfig
    from .client import LeniaClient
    from .config_manager import ConfigManager, load_config
except ImportError:
    from core import LeniaField, FieldConfig
    from client import LeniaClient
    from config_manager import ConfigManager, load_config


class MetricsLogger:
    """
    Logger for Lenia simulation metrics.

    Tracks field, resource, injection, and growth statistics over time.
    Can output to console and/or CSV file for later analysis.
    """

    def __init__(self, log_file: Optional[str] = None, buffer_size: int = 100):
        """
        Initialize metrics logger.

        Args:
            log_file: Optional path to CSV file for logging
            buffer_size: Number of recent metrics to keep in memory
        """
        self.log_file = log_file
        self.buffer_size = buffer_size
        self._buffer: deque = deque(maxlen=buffer_size)
        self._csv_writer = None
        self._csv_file = None
        self._step_count = 0
        self._enabled = False
        self._console_interval = 30  # Print to console every N steps

        if log_file:
            self._setup_csv(log_file)

    def _setup_csv(self, path: str):
        """Setup CSV file for logging."""
        self._csv_file = open(path, 'w', newline='')
        self._csv_writer = None  # Will be created on first write with headers

    def enable(self):
        """Enable logging."""
        self._enabled = True
        print("[MetricsLogger] Enabled")

    def disable(self):
        """Disable logging."""
        self._enabled = False
        print("[MetricsLogger] Disabled")

    def toggle(self):
        """Toggle logging on/off."""
        if self._enabled:
            self.disable()
        else:
            self.enable()

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def log(self, metrics: Dict[str, Any]):
        """
        Log metrics for current step.

        Args:
            metrics: Dictionary of metrics from LeniaField.step_with_metrics()
        """
        if not self._enabled:
            return

        self._step_count += 1
        metrics['step'] = self._step_count

        # Add timestamp
        metrics['timestamp'] = time.time()

        # Remove the field array from logged metrics (too large)
        log_metrics = {k: v for k, v in metrics.items() if k != 'field'}

        # Add to buffer
        self._buffer.append(log_metrics)

        # Write to CSV
        if self._csv_file:
            if self._csv_writer is None:
                # Create writer with headers on first write
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=log_metrics.keys())
                self._csv_writer.writeheader()
            self._csv_writer.writerow(log_metrics)
            self._csv_file.flush()

        # Print to console periodically
        if self._step_count % self._console_interval == 0:
            self._print_summary(log_metrics)

    def _print_summary(self, metrics: Dict[str, Any]):
        """Print summary to console."""
        print(f"\n[Step {metrics['step']}] Lenia Metrics:")
        print(f"  Field:    mean={metrics.get('field_mean', 0):.4f}, max={metrics.get('field_max', 0):.4f}, std={metrics.get('field_std', 0):.4f}")

        if 'resource_mean' in metrics:
            print(f"  Resource: mean={metrics.get('resource_mean', 0):.4f}, min={metrics.get('resource_min', 0):.4f}, max={metrics.get('resource_max', 0):.4f}")

        print(f"  Injection: total={metrics.get('injection_total', 0):.2f}, max={metrics.get('injection_max', 0):.4f}")

        print(f"  Growth:   pos={metrics.get('growth_positive', 0):.2f}, neg={metrics.get('growth_negative', 0):.2f}, mean={metrics.get('growth_mean', 0):.4f}")
        print(f"  Decay:    total={metrics.get('decay_total', 0):.2f}")
        print(f"  Delta:    mean={metrics.get('delta_field_mean', 0):.6f}, max={metrics.get('delta_field_max', 0):.4f}")

        if 'delta_resource_mean' in metrics:
            print(f"  Res.Delta: mean={metrics.get('delta_resource_mean', 0):.6f}")

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent metrics."""
        return list(self._buffer)[-n:]

    def get_averages(self, n: int = 30) -> Dict[str, float]:
        """Get averages over last n steps."""
        if len(self._buffer) == 0:
            return {}

        recent = list(self._buffer)[-n:]
        if not recent:
            return {}

        # Compute averages for numeric fields
        averages = {}
        keys = [k for k in recent[0].keys() if k not in ('step', 'timestamp', 'field')]

        for key in keys:
            values = [m.get(key, 0) for m in recent if key in m]
            if values:
                averages[key] = sum(values) / len(values)

        return averages

    def close(self):
        """Close log file."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    # Movement
    speed: float = 2.0                    # Base movement speed (pixels/step)
    turn_rate: float = 0.3                # Max angular change per step (radians)
    persistence: float = 0.8              # Velocity correlation (0=random, 1=straight)

    # Boundaries
    boundary_mode: str = "reflect"        # "reflect" or "wrap"
    boundary_margin: float = 5.0          # Distance from edge for boundary response

    # Injection (must be > 1.0 to overcome negative Lenia growth on empty field)
    injection_power: float = 2.0          # Morphogen excretion rate
    injection_enabled: bool = True        # Whether agent injects into field

    # Visual
    color: Tuple[int, int, int] = (255, 255, 0)  # BGR color for rendering (cyan - contrasts with PLASMA)
    radius: int = 4                       # Display radius in pixels


class Agent:
    """
    Single agent performing directed random walk with persistence.

    Implements correlated random walk (CRW) where:
    - Direction changes gradually (controlled by turn_rate)
    - Speed varies around base speed
    - Velocity has temporal correlation (persistence)
    """

    def __init__(self,
                 x: float,
                 y: float,
                 config: Optional[AgentConfig] = None,
                 agent_id: int = 0,
                 initial_heading: Optional[float] = None):
        """
        Initialize agent at position.

        Args:
            x: Initial x position
            y: Initial y position
            config: Agent behavior configuration
            agent_id: Unique identifier for this agent
            initial_heading: Initial direction in radians (None = random)
        """
        self.config = config or AgentConfig()
        self.id = agent_id

        # Position state
        self._position = np.array([x, y], dtype=np.float32)

        # Heading in radians [0, 2*pi)
        if initial_heading is None:
            self._heading = np.random.uniform(0, 2 * np.pi)
        else:
            self._heading = initial_heading

        # Current speed (varies around base speed)
        self._speed = self.config.speed

        # For statistics
        self._total_distance = 0.0
        self._step_count = 0

    @property
    def position(self) -> np.ndarray:
        """Current position as [x, y] array."""
        return self._position.copy()

    @property
    def heading(self) -> float:
        """Current heading in radians."""
        return self._heading

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity vector [vx, vy]."""
        return np.array([
            np.cos(self._heading) * self._speed,
            np.sin(self._heading) * self._speed
        ])

    def update(self, field_bounds: Tuple[int, int]) -> np.ndarray:
        """
        Update agent position for one time step.

        Args:
            field_bounds: (width, height) of the field

        Returns:
            New position as [x, y] array
        """
        # Update heading with persistence-weighted random turn
        self._update_heading()

        # Update speed with small random variation
        self._update_speed()

        # Calculate new position
        dx = np.cos(self._heading) * self._speed
        dy = np.sin(self._heading) * self._speed
        new_pos = self._position + np.array([dx, dy])

        # Handle boundaries
        new_pos, self._heading = self._handle_boundary(
            new_pos, self._heading, field_bounds
        )

        # Update state
        distance = np.linalg.norm(new_pos - self._position)
        self._total_distance += distance
        self._step_count += 1
        self._position = new_pos

        return self._position.copy()

    def _update_heading(self):
        """Update heading with correlated random walk."""
        # Random turn with persistence
        # Higher persistence = smaller random component
        max_turn = self.config.turn_rate * (1 - self.config.persistence)
        turn = np.random.uniform(-max_turn, max_turn)

        # Add small persistent drift
        drift = np.random.normal(0, 0.1 * self.config.turn_rate)

        self._heading += turn + drift
        self._heading = self._heading % (2 * np.pi)

    def _update_speed(self):
        """Update speed with small random variation."""
        # Speed varies by +/- 20% around base
        noise = np.random.uniform(0.8, 1.2)
        self._speed = self.config.speed * noise

    def _handle_boundary(self,
                        pos: np.ndarray,
                        heading: float,
                        bounds: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """Handle boundary conditions based on config."""
        width, height = bounds
        margin = self.config.boundary_margin

        if self.config.boundary_mode == "wrap":
            # Toroidal wrap
            pos[0] = pos[0] % width
            pos[1] = pos[1] % height

        elif self.config.boundary_mode == "reflect":
            # Reflection with heading reversal
            if pos[0] < margin:
                pos[0] = margin + (margin - pos[0])
                heading = np.pi - heading
            elif pos[0] > width - margin:
                pos[0] = (width - margin) - (pos[0] - (width - margin))
                heading = np.pi - heading

            if pos[1] < margin:
                pos[1] = margin + (margin - pos[1])
                heading = -heading
            elif pos[1] > height - margin:
                pos[1] = (height - margin) - (pos[1] - (height - margin))
                heading = -heading

            # Normalize heading
            heading = heading % (2 * np.pi)

        else:  # clamp
            pos[0] = np.clip(pos[0], 0, width - 1)
            pos[1] = np.clip(pos[1], 0, height - 1)

        return pos, heading

    def set_position(self, x: float, y: float):
        """Manually set agent position."""
        self._position = np.array([x, y], dtype=np.float32)

    def set_heading(self, heading: float):
        """Manually set agent heading."""
        self._heading = heading % (2 * np.pi)

    def get_stats(self) -> Dict[str, float]:
        """Get agent statistics."""
        avg_speed = self._total_distance / max(1, self._step_count)
        return {
            "total_distance": self._total_distance,
            "step_count": self._step_count,
            "average_speed": avg_speed
        }


@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    # Field
    field_config: FieldConfig = field(default_factory=FieldConfig)

    # Agents
    num_agents: int = 10
    agent_config: AgentConfig = field(default_factory=AgentConfig)

    # Backend
    use_client: bool = False              # Use LeniaClient (subprocess) vs direct
    client_port: int = 5560

    # Visualization
    display_scale: int = 1                # Window scale factor
    colormap: int = 13                    # cv2.COLORMAP_PLASMA = 13 (black -> purple -> pink -> orange -> yellow)
    show_fps: bool = True
    show_stats: bool = True
    target_fps: int = 60                  # Target frame rate (0 = unlimited)
    display_skip: int = 1                 # Only render every N frames (1 = every frame)
    display_downscale: int = 1            # Downscale display by this factor (computation stays full res)


class Simulation:
    """
    Main simulation class combining agents with Lenia field.

    Manages multiple agents, steps the Lenia field, and optionally
    handles visualization.
    """

    def __init__(self,
                 config: Optional[SimulationConfig] = None,
                 agents: Optional[List[Agent]] = None):
        """
        Initialize simulation.

        Args:
            config: Simulation configuration
            agents: Optional pre-created agents (overrides config.num_agents)
        """
        self.config = config or SimulationConfig()

        # Initialize field backend
        if self.config.use_client:
            self._client = LeniaClient.spawn(
                port=self.config.client_port,
                width=self.config.field_config.width,
                height=self.config.field_config.height,
                dt=self.config.field_config.dt,
                diffusion=self.config.field_config.diffusion,
                decay=self.config.field_config.decay
            )
            self._field = None
        else:
            self._field = LeniaField(self.config.field_config)
            self._client = None

        # Initialize agents
        if agents is not None:
            self._agents = agents
        else:
            self._agents = self._create_agents()

        # State
        self._running = False
        self._paused = False
        self._step_count = 0
        self._last_field: Optional[np.ndarray] = None

        # FPS tracking
        self._fps = 0.0

        # Metrics logging (disabled by default, toggle with 'm' key)
        self._metrics_logger: Optional[MetricsLogger] = None
        self._use_metrics = False

    def _create_agents(self) -> List[Agent]:
        """Create agents at random positions."""
        width = self.config.field_config.width
        height = self.config.field_config.height
        agents = []

        for i in range(self.config.num_agents):
            x = np.random.uniform(20, width - 20)
            y = np.random.uniform(20, height - 20)
            agent = Agent(
                x=x, y=y,
                config=self.config.agent_config,
                agent_id=i
            )
            agents.append(agent)

        return agents

    @property
    def agents(self) -> List[Agent]:
        """List of all agents."""
        return self._agents

    @property
    def field(self) -> Optional[np.ndarray]:
        """Current field state."""
        return self._last_field

    @property
    def fps(self) -> float:
        """Current frames per second."""
        return self._fps

    @fps.setter
    def fps(self, value: float):
        """Set FPS value."""
        self._fps = value

    @property
    def is_paused(self) -> bool:
        """Whether simulation is paused."""
        return self._paused

    def add_agent(self, x: float, y: float,
                  config: Optional[AgentConfig] = None) -> Agent:
        """Add a new agent at position."""
        agent = Agent(
            x=x, y=y,
            config=config or self.config.agent_config,
            agent_id=len(self._agents)
        )
        self._agents.append(agent)
        return agent

    def remove_agent(self, agent_id: int) -> bool:
        """Remove agent by ID."""
        for i, agent in enumerate(self._agents):
            if agent.id == agent_id:
                self._agents.pop(i)
                return True
        return False

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance simulation by one step.

        Returns:
            Tuple of (field, positions) where:
                - field: (H, W) float32 array
                - positions: (N, 2) array of agent positions
        """
        bounds = (self.config.field_config.width,
                  self.config.field_config.height)

        # Update all agents
        positions = []
        powers = []
        for agent in self._agents:
            pos = agent.update(bounds)
            positions.append(pos)
            if agent.config.injection_enabled:
                powers.append(agent.config.injection_power)
            else:
                powers.append(0.0)

        positions = np.array(positions, dtype=np.float32) if positions else np.zeros((0, 2), dtype=np.float32)
        powers = np.array(powers, dtype=np.float32) if powers else np.zeros(0, dtype=np.float32)

        # Filter out agents with zero power
        if len(powers) > 0:
            mask = powers > 0
            inject_positions = positions[mask] if np.any(mask) else None
            inject_powers = powers[mask] if np.any(mask) else None
        else:
            inject_positions = None
            inject_powers = None

        # Step field (with or without metrics)
        if self._client is not None:
            response = self._client.step(
                positions=inject_positions,
                powers=inject_powers
            )
            self._last_field = response.field if response.ok else self._last_field
        elif self._use_metrics and self._metrics_logger:
            # Use step_with_metrics for logging
            metrics = self._field.step_with_metrics(
                positions=inject_positions,
                powers=inject_powers
            )
            self._last_field = metrics['field']
            self._metrics_logger.log(metrics)
        else:
            self._last_field = self._field.step(
                positions=inject_positions,
                powers=inject_powers
            )

        self._step_count += 1
        return self._last_field, positions

    def reset(self):
        """Reset simulation to initial state."""
        # Reset field
        if self._client is not None:
            self._client.reset()
        else:
            self._field.reset()

        # Reset agents to random positions
        self._agents = self._create_agents()
        self._step_count = 0
        self._last_field = None

    def pause(self):
        """Pause simulation."""
        self._paused = True

    def resume(self):
        """Resume simulation."""
        self._paused = False

    def toggle_pause(self):
        """Toggle pause state."""
        self._paused = not self._paused

    def get_positions(self) -> np.ndarray:
        """Get all agent positions as (N, 2) array."""
        if not self._agents:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array([a.position for a in self._agents], dtype=np.float32)

    def get_stats(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        return {
            "step_count": self._step_count,
            "num_agents": len(self._agents),
            "fps": self._fps,
            "field_max": float(np.max(self._last_field)) if self._last_field is not None else 0,
            "field_mean": float(np.mean(self._last_field)) if self._last_field is not None else 0,
        }

    def update_field_config(self, **kwargs):
        """Update field configuration at runtime."""
        if self._client is not None:
            self._client.config(**kwargs)
        else:
            self._field.update_config(**kwargs)

    def enable_metrics(self, log_file: Optional[str] = None):
        """
        Enable metrics logging.

        Args:
            log_file: Optional path to CSV file for logging metrics
        """
        if self._metrics_logger is None:
            self._metrics_logger = MetricsLogger(log_file=log_file)
        self._metrics_logger.enable()
        self._use_metrics = True

    def disable_metrics(self):
        """Disable metrics logging."""
        if self._metrics_logger:
            self._metrics_logger.disable()
        self._use_metrics = False

    def toggle_metrics(self):
        """Toggle metrics logging on/off."""
        if self._use_metrics:
            self.disable_metrics()
        else:
            self.enable_metrics()
        return self._use_metrics

    @property
    def metrics_enabled(self) -> bool:
        """Whether metrics logging is enabled."""
        return self._use_metrics

    @property
    def metrics_logger(self) -> Optional[MetricsLogger]:
        """Get the metrics logger instance."""
        return self._metrics_logger

    def get_resource(self) -> Optional[np.ndarray]:
        """Get current resource field (Suzuki mode only)."""
        if self._field is not None and self.config.field_config.suzuki_style:
            return self._field.get_resource()
        return None

    def close(self):
        """Clean up resources."""
        if self._client is not None:
            try:
                self._client.shutdown()
            except Exception:
                pass
        if self._metrics_logger:
            self._metrics_logger.close()


class SimulationVisualizer:
    """
    OpenCV-based visualizer for the simulation.

    Handles rendering, keyboard input, and mouse interaction.
    """

    # Key mappings
    KEY_QUIT = ord('q')
    KEY_RESET = ord('r')
    KEY_PAUSE = ord(' ')  # Space
    KEY_ADD_AGENT = ord('a')
    KEY_HELP = ord('h')
    KEY_METRICS = ord('m')
    KEY_RESOURCE = ord('v')  # Toggle resource view

    def __init__(self,
                 simulation: Simulation,
                 window_name: str = "Lenia Simulation"):
        """
        Initialize visualizer.

        Args:
            simulation: Simulation instance to visualize
            window_name: Name for the OpenCV window
        """
        self.sim = simulation
        self.window_name = window_name
        self.config = simulation.config

        # Window dimensions
        self.width = self.config.field_config.width * self.config.display_scale
        self.height = self.config.field_config.height * self.config.display_scale

        # Mouse state
        self._mouse_pos: Optional[Tuple[int, int]] = None
        self._adding_agents = False

        # FPS tracking
        self._frame_times: List[float] = []

        # Track current config values for display
        self._current_diffusion = self.config.field_config.diffusion
        self._current_decay = self.config.field_config.decay

        # Resource view toggle (for Suzuki mode)
        self._show_resource = False

    def _setup_window(self):
        """Create and configure OpenCV window."""
        import cv2
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        import cv2
        # Scale coordinates back to field space
        scale = self.config.display_scale
        fx, fy = x // scale, y // scale

        if event == cv2.EVENT_MOUSEMOVE:
            self._mouse_pos = (fx, fy)

        elif event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_pos = (fx, fy)
            if self._adding_agents:
                self.sim.add_agent(fx, fy)
                print(f"Added agent at ({fx}, {fy})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click: remove nearest agent
            self._remove_nearest_agent(fx, fy)

    def _remove_nearest_agent(self, x: float, y: float):
        """Remove agent nearest to click position."""
        if not self.sim.agents:
            return

        positions = self.sim.get_positions()
        if len(positions) == 0:
            return

        distances = np.sqrt(np.sum((positions - [x, y])**2, axis=1))
        nearest_idx = np.argmin(distances)

        if distances[nearest_idx] < 20:  # Only if close enough
            agent_id = self.sim.agents[nearest_idx].id
            self.sim.remove_agent(agent_id)
            print(f"Removed agent {agent_id}")

    def _render_frame(self, field: np.ndarray,
                      positions: np.ndarray) -> np.ndarray:
        """Render a single frame."""
        import cv2

        # Choose what to display: field or resource
        if self._show_resource and self.config.field_config.suzuki_style:
            resource = self.sim.get_resource()
            if resource is not None:
                # Normalize resource to [0, 1] range based on r_max
                display_data = resource / self.config.field_config.r_max
            else:
                display_data = field
        else:
            display_data = field

        # Apply colormap
        data_uint8 = (np.clip(display_data, 0, 1) * 255).astype(np.uint8)
        frame = cv2.applyColorMap(data_uint8, self.config.colormap)

        # Scale if needed
        if self.config.display_scale > 1:
            frame = cv2.resize(
                frame, None,
                fx=self.config.display_scale,
                fy=self.config.display_scale,
                interpolation=cv2.INTER_NEAREST
            )

        # Draw agents
        scale = self.config.display_scale
        for i, agent in enumerate(self.sim.agents):
            if i < len(positions):
                pos = positions[i]
                x = int(pos[0] * scale)
                y = int(pos[1] * scale)
                color = agent.config.color
                radius = agent.config.radius * scale

                # Draw agent circle
                cv2.circle(frame, (x, y), radius, color, -1)

                # Draw heading indicator
                heading = agent.heading
                hx = int(x + np.cos(heading) * radius * 2)
                hy = int(y + np.sin(heading) * radius * 2)
                cv2.line(frame, (x, y), (hx, hy), color, 1)

        # Draw overlay text
        if self.config.show_fps or self.config.show_stats:
            self._draw_overlay(frame)

        return frame

    def _draw_overlay(self, frame: np.ndarray):
        """Draw FPS and stats overlay."""
        import cv2

        stats = self.sim.get_stats()

        y_offset = 25
        texts = []

        if self.config.show_fps:
            texts.append(f"FPS: {self.sim.fps:.1f}")

        if self.config.show_stats:
            texts.append(f"Agents: {stats['num_agents']}")
            texts.append(f"Steps: {stats['step_count']}")
            texts.append(f"Field max: {stats['field_max']:.3f}")

        if self.sim.is_paused:
            texts.insert(0, "[PAUSED]")

        if self._adding_agents:
            texts.append("[Click to add agent]")

        if self.sim.metrics_enabled:
            texts.append("[METRICS ON - m to toggle]")

        if self._show_resource:
            texts.append("[RESOURCE VIEW - v to toggle]")

        for text in texts:
            cv2.putText(
                frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            y_offset += 20

    def _handle_key(self, key: int) -> bool:
        """
        Handle keyboard input.

        Returns:
            False if should quit, True otherwise
        """
        if key == self.KEY_QUIT:
            return False

        elif key == self.KEY_RESET:
            self.sim.reset()
            print("Simulation reset")

        elif key == self.KEY_PAUSE:
            self.sim.toggle_pause()
            print("Paused" if self.sim.is_paused else "Resumed")

        elif key == self.KEY_ADD_AGENT:
            self._adding_agents = not self._adding_agents
            print(f"Add agent mode: {'ON' if self._adding_agents else 'OFF'}")

        elif key == self.KEY_HELP:
            self._print_help()

        # Number keys 1-9: adjust injection power
        elif ord('1') <= key <= ord('9'):
            power = (key - ord('0')) * 0.05
            for agent in self.sim.agents:
                agent.config.injection_power = power
            print(f"Injection power: {power:.2f}")

        # d/D: decrease/increase diffusion
        elif key == ord('d'):
            self._current_diffusion = max(0.001, self._current_diffusion - 0.002)
            self.sim.update_field_config(diffusion=self._current_diffusion)
            print(f"Diffusion: {self._current_diffusion:.4f}")
        elif key == ord('D'):
            self._current_diffusion = min(0.1, self._current_diffusion + 0.002)
            self.sim.update_field_config(diffusion=self._current_diffusion)
            print(f"Diffusion: {self._current_diffusion:.4f}")

        # c/C: decrease/increase decay
        elif key == ord('c'):
            self._current_decay = max(0.0001, self._current_decay - 0.001)
            self.sim.update_field_config(decay=self._current_decay)
            print(f"Decay: {self._current_decay:.4f}")
        elif key == ord('C'):
            self._current_decay = min(0.1, self._current_decay + 0.001)
            self.sim.update_field_config(decay=self._current_decay)
            print(f"Decay: {self._current_decay:.4f}")

        # s/S: decrease/increase agent speed
        elif key == ord('s'):
            for agent in self.sim.agents:
                agent.config.speed = max(0.5, agent.config.speed - 0.5)
            print("Agent speed decreased")
        elif key == ord('S'):
            for agent in self.sim.agents:
                agent.config.speed = min(10, agent.config.speed + 0.5)
            print("Agent speed increased")

        # m: toggle metrics logging
        elif key == self.KEY_METRICS:
            enabled = self.sim.toggle_metrics()
            print(f"Metrics logging: {'ON' if enabled else 'OFF'}")

        # v: toggle resource view (Suzuki mode)
        elif key == self.KEY_RESOURCE:
            if self.config.field_config.suzuki_style:
                self._show_resource = not self._show_resource
                print(f"Showing: {'Resource field' if self._show_resource else 'Morphogen field'}")
            else:
                print("Resource view only available in Suzuki mode")

        return True

    def _print_help(self):
        """Print help text to console."""
        print("\n=== Lenia Simulation Controls ===")
        print("  SPACE: Pause/Resume")
        print("  r: Reset simulation")
        print("  a: Toggle add-agent mode (click to add)")
        print("  Right-click: Remove nearest agent")
        print("  1-9: Set injection power")
        print("  d/D: Decrease/Increase diffusion")
        print("  c/C: Decrease/Increase decay")
        print("  s/S: Decrease/Increase agent speed")
        print("  m: Toggle metrics logging (debug)")
        print("  v: Toggle resource/field view (Suzuki mode)")
        print("  h: Show this help")
        print("  q: Quit")
        print("================================\n")

    def _update_fps(self, frame_time: float):
        """Update FPS calculation."""
        self._frame_times.append(frame_time)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        if self._frame_times:
            self.sim.fps = 1.0 / np.mean(self._frame_times)

    def run(self):
        """Main visualization loop (blocking)."""
        import cv2

        self._setup_window()
        self._print_help()

        target_frame_time = 1.0 / self.config.target_fps if self.config.target_fps > 0 else 0

        try:
            while True:
                frame_start = time.time()

                # Step simulation (unless paused)
                if not self.sim.is_paused:
                    field, positions = self.sim.step()
                else:
                    field = self.sim.field
                    positions = self.sim.get_positions()
                    if field is None:
                        # Initial state before first step
                        field = np.zeros((
                            self.config.field_config.height,
                            self.config.field_config.width
                        ))

                # Render
                frame = self._render_frame(field, positions)

                # Display
                cv2.imshow(self.window_name, frame)

                # Handle input
                elapsed = time.time() - frame_start
                wait_time = max(1, int((target_frame_time - elapsed) * 1000))
                key = cv2.waitKey(wait_time) & 0xFF

                if key != 255 and not self._handle_key(key):
                    break

                # Update FPS
                self._update_fps(time.time() - frame_start)

        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up resources."""
        import cv2
        cv2.destroyAllWindows()
        self.sim.close()
        print("Simulation ended")


def run_simulation(
    num_agents: int = 10,
    width: int = 512,
    height: int = 512,
    use_client: bool = False,
    **kwargs
) -> None:
    """
    Run an interactive Lenia simulation with agents.

    This is the main entry point for quick demos.

    Args:
        num_agents: Number of agents to simulate
        width: Field width in pixels
        height: Field height in pixels
        use_client: Use subprocess server (better for GPU)
        **kwargs: Additional parameters (speed, diffusion, decay, etc.)
    """
    # Extract field config parameters
    field_params = {}
    for key in ['dt', 'diffusion', 'decay', 'injection_radius', 'injection_power',
                'kernel_radius', 'growth_mu', 'growth_sigma', 'skip_diffusion',
                'suzuki_style', 'r_max', 'r_initial', 'R_C', 'R_G', 'resource_rule',
                'resource_effect_rate']:
        if key in kwargs:
            field_params[key] = kwargs.pop(key)

    # Note: suzuki_style is enabled by default (prevents Turing pattern accumulation)
    # Set suzuki_style=False to use classic direct injection mode

    field_config = FieldConfig(
        width=width,
        height=height,
        **field_params
    )

    # Extract agent config parameters
    agent_params = {}
    for key in ['speed', 'turn_rate', 'persistence', 'boundary_mode']:
        if key in kwargs:
            agent_params[key] = kwargs.pop(key)

    agent_config = AgentConfig(**agent_params) if agent_params else AgentConfig()

    config = SimulationConfig(
        field_config=field_config,
        num_agents=num_agents,
        agent_config=agent_config,
        use_client=use_client,
    )

    sim = Simulation(config)
    viz = SimulationVisualizer(sim)
    viz.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Lenia Agent Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python simulation.py
    python simulation.py --config custom.yaml
    python simulation.py --override field__width=1024 --override simulation__num_agents=50
        """
    )
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to custom YAML config file')
    parser.add_argument('--override', '-o', action='append', default=[],
                        help='Runtime override in key=value format (use __ for nesting)')

    args = parser.parse_args()

    # Parse overrides into dict
    overrides = {}
    for override in args.override:
        if '=' in override:
            key, value = override.split('=', 1)
            # Try to parse as number or bool
            try:
                if '.' in value:
                    value = float(value)
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string

            # Convert double underscore to nested dict
            parts = key.split('__')
            current = overrides
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

    # Load config
    config = ConfigManager(config_path=args.config, overrides=overrides if overrides else None)

    # Extract values from config
    field_section = config.get_section('field')
    agent_section = config.get_section('agent')
    sim_section = config.get_section('simulation')
    viz_section = config.get_section('visualization')

    # Build FieldConfig
    field_config = FieldConfig(
        width=field_section.get('width', 512),
        height=field_section.get('height', 512),
        kernel_radius=field_section.get('kernel_radius', 13),
        kernel_sigma=field_section.get('kernel_sigma', 0.5),
        growth_mu=field_section.get('growth_mu', 0.15),
        growth_sigma=field_section.get('growth_sigma', 0.015),
        growth_amplitude=field_section.get('growth_amplitude', 1.0),
        dt=field_section.get('dt', 0.1),
        diffusion=field_section.get('diffusion', 0.01),
        decay=field_section.get('decay', 0.02),
        injection_radius=field_section.get('injection_radius', 5.0),
        injection_power=field_section.get('injection_power', 0.1),
        use_fft=field_section.get('use_fft', True),
        skip_diffusion=field_section.get('skip_diffusion', True),
        # Suzuki-style resource dynamics
        suzuki_style=field_section.get('suzuki_style', True),
        r_max=field_section.get('r_max', 1.0),
        r_initial=field_section.get('r_initial', 1.0),
        R_C=field_section.get('R_C', 0.005),
        R_G=field_section.get('R_G', 0.001),
        resource_rule=field_section.get('resource_rule', 'S'),
        agents_emit_resources=field_section.get('agents_emit_resources', True),
        resource_effect_rate=field_section.get('resource_effect_rate', 0.5),
        resource_diffusion=field_section.get('resource_diffusion', 0.1),
    )

    # Build AgentConfig
    agent_color = agent_section.get('color', [255, 255, 0])
    if isinstance(agent_color, list):
        agent_color = tuple(agent_color)

    agent_config = AgentConfig(
        speed=agent_section.get('speed', 2.0),
        turn_rate=agent_section.get('turn_rate', 0.3),
        persistence=agent_section.get('persistence', 0.8),
        boundary_mode=agent_section.get('boundary_mode', 'reflect'),
        boundary_margin=agent_section.get('boundary_margin', 5.0),
        injection_power=agent_section.get('injection_power', 2.0),
        injection_enabled=agent_section.get('injection_enabled', True),
        color=agent_color,
        radius=agent_section.get('radius', 4),
    )

    # Build SimulationConfig
    sim_config = SimulationConfig(
        field_config=field_config,
        agent_config=agent_config,
        num_agents=sim_section.get('num_agents', 10),
        use_client=sim_section.get('use_client', False),
        client_port=sim_section.get('client_port', 5560),
        target_fps=viz_section.get('target_fps', 60),
        display_scale=viz_section.get('display_scale', 1),
        colormap=viz_section.get('colormap', 13),
        show_fps=viz_section.get('show_fps', True),
        show_stats=viz_section.get('show_stats', True),
    )

    # Run simulation
    sim = Simulation(sim_config)
    viz = SimulationVisualizer(sim)
    viz.run()
