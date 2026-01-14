"""
lenia_field - Real-time Lenia morphogen field with external stimulus injection.

Usage:
    from lenia_field import LeniaClient, LeniaField, FieldConfig
    
    # Run as subprocess (recommended for non-blocking)
    client = LeniaClient.spawn(width=512, height=512)
    response = client.step(positions=[[100, 100], [200, 200]])
    field = response.field
    
    # Or run directly (blocking)
    field = LeniaField(FieldConfig(width=512, height=512))
    result = field.step(positions=np.array([[100, 100]]))
"""

from .core import LeniaField, FieldConfig, make_kernel
from .client import LeniaClient, LeniaClientAsync, LeniaResponse
from .config_manager import ConfigManager, load_config
from .simulation import (
    Agent,
    AgentConfig,
    MetricsLogger,
    Simulation,
    SimulationConfig,
    SimulationVisualizer,
    run_simulation,
)
from .optimizer import (
    LeniaOptimizer,
    OptimizationTarget,
    OptimizationResult,
    EvaluationMetrics,
    ParameterSpec,
    quick_optimize,
    optimize_from_simulation,
    plot_sensitivity,
    plot_sensitivity_heatmap,
)
from .presets import (
    PresetManager,
    get_optimized_config,
    apply_optimized_params,
    save_optimized_preset,
    list_presets,
    get_preset_path,
    preset_exists,
    delete_preset,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "LeniaField",
    "FieldConfig",
    "make_kernel",
    # Client
    "LeniaClient",
    "LeniaClientAsync",
    "LeniaResponse",
    # Config
    "ConfigManager",
    "load_config",
    # Simulation
    "Agent",
    "AgentConfig",
    "Simulation",
    "SimulationConfig",
    "SimulationVisualizer",
    "run_simulation",
    # Optimizer
    "LeniaOptimizer",
    "OptimizationTarget",
    "OptimizationResult",
    "EvaluationMetrics",
    "ParameterSpec",
    "quick_optimize",
    "optimize_from_simulation",
    "plot_sensitivity",
    "plot_sensitivity_heatmap",
    # Presets
    "PresetManager",
    "get_optimized_config",
    "apply_optimized_params",
    "save_optimized_preset",
    "list_presets",
    "get_preset_path",
    "preset_exists",
    "delete_preset",
]
