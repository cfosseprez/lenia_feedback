"""
Configuration manager for Lenia Field simulation.

Loads defaults from YAML and allows runtime overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, fields, asdict


def _get_default_config_path() -> Path:
    """Get path to default config file in package."""
    return Path(__file__).parent / "config" / "default.yaml"


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ConfigManager:
    """
    Manages configuration with YAML defaults and runtime overrides.

    Usage:
        # Load defaults
        config = ConfigManager()

        # Load defaults + custom file
        config = ConfigManager(config_path="my_config.yaml")

        # Load defaults + runtime overrides
        config = ConfigManager(overrides={"field": {"width": 1024}})

        # Access values
        width = config.get("field.width")
        field_config = config.get("field")
    """

    _instance: Optional['ConfigManager'] = None

    def __init__(self,
                 config_path: Optional[str] = None,
                 overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize config manager.

        Args:
            config_path: Optional path to custom YAML config file
            overrides: Optional dict of runtime overrides
        """
        # Load default config
        self._config = self._load_yaml(_get_default_config_path())

        # Merge custom config file if provided
        if config_path:
            custom_config = self._load_yaml(Path(config_path))
            self._config = _deep_merge(self._config, custom_config)

        # Apply runtime overrides
        if overrides:
            self._config = _deep_merge(self._config, overrides)

    @classmethod
    def get_instance(cls, **kwargs) -> 'ConfigManager':
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (useful for testing)."""
        cls._instance = None

    def _load_yaml(self, path: Path) -> Dict:
        """Load YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by dot-separated key.

        Args:
            key: Dot-separated key like "field.width" or "agent.speed"
            default: Default value if key not found

        Returns:
            Config value or default
        """
        parts = key.split('.')
        value = self._config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set config value by dot-separated key.

        Args:
            key: Dot-separated key like "field.width"
            value: Value to set
        """
        parts = key.split('.')
        config = self._config

        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]

        config[parts[-1]] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire config section.

        Args:
            section: Section name like "field", "agent", "simulation"

        Returns:
            Dict of section config or empty dict
        """
        return self._config.get(section, {}).copy()

    def to_dict(self) -> Dict[str, Any]:
        """Return full config as dictionary."""
        return self._config.copy()

    def save(self, path: str):
        """Save current config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def __repr__(self):
        return f"ConfigManager({self._config})"


def load_config(config_path: Optional[str] = None,
                **overrides) -> ConfigManager:
    """
    Convenience function to load config with overrides.

    Args:
        config_path: Optional path to custom YAML config
        **overrides: Key-value overrides (use double underscore for nesting)
                    e.g., field__width=1024 becomes {"field": {"width": 1024}}

    Returns:
        ConfigManager instance
    """
    # Convert double-underscore keys to nested dict
    override_dict = {}
    for key, value in overrides.items():
        if value is None:
            continue
        parts = key.split('__')
        current = override_dict
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    return ConfigManager(config_path=config_path, overrides=override_dict if override_dict else None)
