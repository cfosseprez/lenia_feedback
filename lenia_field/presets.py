"""
Preset management for optimized Lenia field parameters.

Presets are stored as YAML files and can be loaded to create FieldConfig instances.
This allows you to save optimized parameters and reuse them across different
applications (LeniaField, LeniaClient, Simulation, custom apps).

Usage:
    from lenia_field import get_optimized_config, list_presets

    # List available presets
    print(list_presets())  # ['optimized', 'my_preset', ...]

    # Load optimized config
    config = get_optimized_config()  # Uses 'optimized' preset
    config = get_optimized_config("my_preset")  # Uses custom preset

    # Use with any component
    field = LeniaField(config)
    sim = Simulation(SimulationConfig(field_config=config))
"""

import json
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .core import FieldConfig


# =============================================================================
# Constants
# =============================================================================

# Default presets directory inside the package
PRESETS_DIR = Path(__file__).parent / "config" / "presets"
DEFAULT_PRESET_NAME = "optimized"

# Parameters that can be optimized (and stored in presets)
OPTIMIZABLE_PARAMS = [
    "growth_mu",
    "growth_sigma",
    "growth_amplitude",
    "dt",
    "decay",
    "injection_power",
    "injection_radius",
    # Suzuki mode parameters
    "r_max",
    "R_C",
    "R_G",
    "resource_effect_rate",
]


# =============================================================================
# PresetManager Class
# =============================================================================


class PresetManager:
    """
    Manage optimized parameter presets.

    Presets are stored as YAML files in the presets directory.
    Each preset contains:
    - params: Dict of optimized parameter values
    - metadata: Optional info about when/how the preset was created
    """

    _custom_dir: Optional[Path] = None

    @classmethod
    def set_presets_dir(cls, path: Union[str, Path]) -> None:
        """Set a custom presets directory."""
        cls._custom_dir = Path(path)
        cls._custom_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_presets_dir(cls) -> Path:
        """Get the presets directory path."""
        if cls._custom_dir is not None:
            return cls._custom_dir
        return PRESETS_DIR

    @classmethod
    def list_presets(cls) -> List[str]:
        """
        List all available preset names.

        Returns:
            List of preset names (without .yaml extension)
        """
        presets_dir = cls.get_presets_dir()
        if not presets_dir.exists():
            return []

        presets = []
        for f in presets_dir.glob("*.yaml"):
            presets.append(f.stem)
        for f in presets_dir.glob("*.yml"):
            if f.stem not in presets:
                presets.append(f.stem)
        for f in presets_dir.glob("*.json"):
            if f.stem not in presets:
                presets.append(f.stem)

        return sorted(presets)

    @classmethod
    def preset_exists(cls, name: str) -> bool:
        """Check if a preset exists."""
        presets_dir = cls.get_presets_dir()
        return (
            (presets_dir / f"{name}.yaml").exists()
            or (presets_dir / f"{name}.yml").exists()
            or (presets_dir / f"{name}.json").exists()
        )

    @classmethod
    def get_preset_path(cls, name: str) -> Path:
        """Get the file path for a preset."""
        presets_dir = cls.get_presets_dir()

        # Try different extensions
        for ext in [".yaml", ".yml", ".json"]:
            path = presets_dir / f"{name}{ext}"
            if path.exists():
                return path

        # Default to .yaml for new presets
        return presets_dir / f"{name}.yaml"

    @classmethod
    def load_preset(cls, name: str = DEFAULT_PRESET_NAME) -> Dict[str, Any]:
        """
        Load a preset from file.

        Args:
            name: Preset name (without extension)

        Returns:
            Dict with 'params' and optional 'metadata' keys

        Raises:
            FileNotFoundError: If preset doesn't exist
        """
        path = cls.get_preset_path(name)

        if not path.exists():
            available = cls.list_presets()
            raise FileNotFoundError(
                f"Preset '{name}' not found at {path}. "
                f"Available presets: {available or 'none'}"
            )

        if path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
        else:
            with open(path, "r") as f:
                data = yaml.safe_load(f)

        # Handle both old format (just params) and new format (params + metadata)
        if "params" not in data:
            # Old format: entire file is params
            return {"params": data, "metadata": {}}

        return data

    @classmethod
    def save_preset(
        cls,
        params: Dict[str, float],
        name: str = DEFAULT_PRESET_NAME,
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "yaml",
    ) -> Path:
        """
        Save parameters as a preset.

        Args:
            params: Dict of parameter values
            name: Preset name
            metadata: Optional metadata (fitness, date, etc.)
            format: File format ('yaml' or 'json')

        Returns:
            Path to saved preset file
        """
        presets_dir = cls.get_presets_dir()
        presets_dir.mkdir(parents=True, exist_ok=True)

        # Filter to only optimizable params
        filtered_params = {
            k: float(v) for k, v in params.items() if k in OPTIMIZABLE_PARAMS
        }

        data = {
            "params": filtered_params,
            "metadata": metadata or {
                "created": datetime.now().isoformat(),
            },
        }

        ext = ".json" if format == "json" else ".yaml"
        path = presets_dir / f"{name}{ext}"

        if format == "json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        return path

    @classmethod
    def delete_preset(cls, name: str) -> bool:
        """
        Delete a preset.

        Args:
            name: Preset name

        Returns:
            True if deleted, False if not found
        """
        path = cls.get_preset_path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    @classmethod
    def get_field_config(
        cls,
        name: str = DEFAULT_PRESET_NAME,
        **overrides,
    ) -> FieldConfig:
        """
        Load a preset and return a FieldConfig.

        Args:
            name: Preset name
            **overrides: Additional parameters to override

        Returns:
            FieldConfig with optimized parameters
        """
        preset_data = cls.load_preset(name)
        params = preset_data["params"].copy()
        params.update(overrides)

        return FieldConfig(**params)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_optimized_config(
    preset: str = DEFAULT_PRESET_NAME,
    **overrides,
) -> FieldConfig:
    """
    Load optimized parameters and return a FieldConfig.

    This is the main function to get a config with optimized parameters.
    The returned config can be used with any component:
    - LeniaField(config)
    - Simulation(SimulationConfig(field_config=config))
    - LeniaClient.spawn(...config parameters...)

    Args:
        preset: Name of the preset to load (default: "optimized")
        **overrides: Override specific parameters (e.g., width=1024)

    Returns:
        FieldConfig with optimized parameters

    Example:
        # Basic usage
        config = get_optimized_config()
        field = LeniaField(config)

        # With overrides
        config = get_optimized_config(width=1024, height=1024)

        # Custom preset
        config = get_optimized_config("my_preset")
    """
    return PresetManager.get_field_config(preset, **overrides)


def apply_optimized_params(
    config: FieldConfig,
    preset: str = DEFAULT_PRESET_NAME,
) -> FieldConfig:
    """
    Apply optimized parameters to an existing FieldConfig.

    This preserves non-optimized parameters (like width, height) from
    the original config while updating the optimized parameters.

    Args:
        config: Existing FieldConfig to update
        preset: Name of the preset to apply

    Returns:
        New FieldConfig with optimized parameters applied

    Example:
        # Create config with custom size
        config = FieldConfig(width=1024, height=1024)

        # Apply optimized params (preserves width/height)
        config = apply_optimized_params(config)
    """
    preset_data = PresetManager.load_preset(preset)
    optimized_params = preset_data["params"]

    # Get current config as dict
    config_dict = asdict(config)

    # Update only the optimized parameters
    for key, value in optimized_params.items():
        if key in config_dict:
            config_dict[key] = value

    return FieldConfig(**config_dict)


def save_optimized_preset(
    result: "OptimizationResult",
    name: str = DEFAULT_PRESET_NAME,
    format: str = "yaml",
) -> Path:
    """
    Save optimization result as a preset.

    Args:
        result: OptimizationResult from optimizer
        name: Preset name
        format: File format ('yaml' or 'json')

    Returns:
        Path to saved preset file

    Example:
        result = optimizer.optimize(trajectories)
        save_optimized_preset(result, "my_optimized")
    """
    # Import here to avoid circular import
    from .optimizer import OptimizationResult

    if not isinstance(result, OptimizationResult):
        raise TypeError(f"Expected OptimizationResult, got {type(result)}")

    metadata = {
        "created": datetime.now().isoformat(),
        "fitness": result.best_fitness,
        "metrics": result.best_metrics,
        "elapsed_time": result.elapsed_time,
        "config": result.config,
    }

    return PresetManager.save_preset(
        result.best_params,
        name=name,
        metadata=metadata,
        format=format,
    )


def list_presets() -> List[str]:
    """
    List all available preset names.

    Returns:
        List of preset names

    Example:
        presets = list_presets()
        print(presets)  # ['optimized', 'my_preset', ...]
    """
    return PresetManager.list_presets()


def get_preset_path(name: str = DEFAULT_PRESET_NAME) -> Path:
    """
    Get the file path for a preset.

    Args:
        name: Preset name

    Returns:
        Path to the preset file
    """
    return PresetManager.get_preset_path(name)


def preset_exists(name: str) -> bool:
    """
    Check if a preset exists.

    Args:
        name: Preset name

    Returns:
        True if preset exists
    """
    return PresetManager.preset_exists(name)


def delete_preset(name: str) -> bool:
    """
    Delete a preset.

    Args:
        name: Preset name

    Returns:
        True if deleted, False if not found
    """
    return PresetManager.delete_preset(name)
