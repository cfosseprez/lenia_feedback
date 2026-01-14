"""Tests for presets.py module."""

import tempfile
from pathlib import Path

import pytest

from lenia_field.core import FieldConfig
from lenia_field.presets import (
    DEFAULT_PRESET_NAME,
    OPTIMIZABLE_PARAMS,
    PresetManager,
    apply_optimized_params,
    delete_preset,
    get_optimized_config,
    get_preset_path,
    list_presets,
    preset_exists,
    save_optimized_preset,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_presets_dir(tmp_path):
    """Create a temporary presets directory."""
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    PresetManager.set_presets_dir(presets_dir)
    yield presets_dir
    # Reset to default after test
    PresetManager._custom_dir = None


@pytest.fixture
def sample_params():
    """Sample optimized parameters."""
    return {
        "growth_mu": 0.18,
        "growth_sigma": 0.02,
        "growth_amplitude": 1.2,
        "dt": 0.12,
        "decay": 0.03,
        "injection_power": 0.15,
        "injection_radius": 6.0,
        "r_max": 1.5,
        "R_C": 0.008,
        "R_G": 0.002,
        "resource_effect_rate": 0.12,
    }


@pytest.fixture
def mock_optimization_result(sample_params):
    """Create a mock OptimizationResult for testing."""
    from lenia_field.optimizer import OptimizationResult

    return OptimizationResult(
        best_params=sample_params,
        best_fitness=15.5,
        best_metrics={"field_mean": 0.15, "field_std": 0.1},
        history=[],
        parameter_specs={},
        target={},
        config={"width": 512, "height": 512},
        elapsed_time=10.0,
    )


# =============================================================================
# PresetManager Tests
# =============================================================================


class TestPresetManagerBasic:
    """Basic PresetManager functionality tests."""

    def test_get_presets_dir_default(self):
        """Test default presets directory."""
        PresetManager._custom_dir = None
        dir_path = PresetManager.get_presets_dir()
        assert dir_path.name == "presets"
        assert "config" in str(dir_path)

    def test_set_presets_dir(self, tmp_path):
        """Test setting custom presets directory."""
        custom_dir = tmp_path / "custom_presets"
        PresetManager.set_presets_dir(custom_dir)
        assert PresetManager.get_presets_dir() == custom_dir
        assert custom_dir.exists()
        PresetManager._custom_dir = None

    def test_list_presets_empty(self, temp_presets_dir):
        """Test listing presets when directory is empty."""
        presets = PresetManager.list_presets()
        assert presets == []

    def test_list_presets_with_files(self, temp_presets_dir):
        """Test listing presets with various file types."""
        (temp_presets_dir / "preset1.yaml").touch()
        (temp_presets_dir / "preset2.yml").touch()
        (temp_presets_dir / "preset3.json").touch()
        (temp_presets_dir / "not_a_preset.txt").touch()

        presets = PresetManager.list_presets()
        assert sorted(presets) == ["preset1", "preset2", "preset3"]


class TestPresetManagerSaveLoad:
    """Save and load functionality tests."""

    def test_save_preset_yaml(self, temp_presets_dir, sample_params):
        """Test saving preset as YAML."""
        path = PresetManager.save_preset(sample_params, "test_preset", format="yaml")

        assert path.exists()
        assert path.suffix == ".yaml"
        assert "test_preset" in str(path)

    def test_save_preset_json(self, temp_presets_dir, sample_params):
        """Test saving preset as JSON."""
        path = PresetManager.save_preset(sample_params, "test_json", format="json")

        assert path.exists()
        assert path.suffix == ".json"

    def test_save_filters_params(self, temp_presets_dir):
        """Test that save filters to only optimizable params."""
        params = {
            "growth_mu": 0.2,
            "unknown_param": 999,
            "width": 1024,  # Not optimizable
        }
        PresetManager.save_preset(params, "filtered")

        data = PresetManager.load_preset("filtered")
        assert "growth_mu" in data["params"]
        assert "unknown_param" not in data["params"]
        assert "width" not in data["params"]

    def test_load_preset(self, temp_presets_dir, sample_params):
        """Test loading a preset."""
        PresetManager.save_preset(sample_params, "load_test")
        data = PresetManager.load_preset("load_test")

        assert "params" in data
        assert "metadata" in data
        assert data["params"]["growth_mu"] == sample_params["growth_mu"]

    def test_load_preset_not_found(self, temp_presets_dir):
        """Test loading non-existent preset raises error."""
        with pytest.raises(FileNotFoundError):
            PresetManager.load_preset("nonexistent")

    def test_preset_exists(self, temp_presets_dir, sample_params):
        """Test preset_exists functionality."""
        assert not PresetManager.preset_exists("check_test")
        PresetManager.save_preset(sample_params, "check_test")
        assert PresetManager.preset_exists("check_test")

    def test_delete_preset(self, temp_presets_dir, sample_params):
        """Test deleting a preset."""
        PresetManager.save_preset(sample_params, "delete_test")
        assert PresetManager.preset_exists("delete_test")

        result = PresetManager.delete_preset("delete_test")
        assert result is True
        assert not PresetManager.preset_exists("delete_test")

    def test_delete_nonexistent_preset(self, temp_presets_dir):
        """Test deleting non-existent preset returns False."""
        result = PresetManager.delete_preset("nonexistent")
        assert result is False

    def test_get_field_config(self, temp_presets_dir, sample_params):
        """Test getting FieldConfig from preset."""
        PresetManager.save_preset(sample_params, "config_test")
        config = PresetManager.get_field_config("config_test")

        assert isinstance(config, FieldConfig)
        assert config.growth_mu == sample_params["growth_mu"]
        assert config.decay == sample_params["decay"]

    def test_get_field_config_with_overrides(self, temp_presets_dir, sample_params):
        """Test getting FieldConfig with parameter overrides."""
        PresetManager.save_preset(sample_params, "override_test")
        config = PresetManager.get_field_config(
            "override_test",
            width=1024,
            height=768,
        )

        assert config.width == 1024
        assert config.height == 768
        assert config.growth_mu == sample_params["growth_mu"]


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_optimized_config(self, temp_presets_dir, sample_params):
        """Test get_optimized_config function."""
        PresetManager.save_preset(sample_params, "optimized")
        config = get_optimized_config()

        assert isinstance(config, FieldConfig)
        assert config.growth_mu == sample_params["growth_mu"]

    def test_get_optimized_config_custom_preset(self, temp_presets_dir, sample_params):
        """Test get_optimized_config with custom preset name."""
        PresetManager.save_preset(sample_params, "custom")
        config = get_optimized_config("custom")

        assert config.growth_mu == sample_params["growth_mu"]

    def test_get_optimized_config_with_overrides(self, temp_presets_dir, sample_params):
        """Test get_optimized_config with overrides."""
        PresetManager.save_preset(sample_params, "optimized")
        config = get_optimized_config(width=2048)

        assert config.width == 2048
        assert config.growth_mu == sample_params["growth_mu"]

    def test_apply_optimized_params(self, temp_presets_dir, sample_params):
        """Test apply_optimized_params function."""
        PresetManager.save_preset(sample_params, "apply_test")

        # Create config with custom size
        original = FieldConfig(width=1024, height=768)
        updated = apply_optimized_params(original, "apply_test")

        # Size should be preserved
        assert updated.width == 1024
        assert updated.height == 768

        # Optimized params should be applied
        assert updated.growth_mu == sample_params["growth_mu"]
        assert updated.decay == sample_params["decay"]

    def test_save_optimized_preset(self, temp_presets_dir, mock_optimization_result):
        """Test save_optimized_preset function."""
        path = save_optimized_preset(mock_optimization_result, "save_test")

        assert path.exists()
        data = PresetManager.load_preset("save_test")
        assert data["params"]["growth_mu"] == mock_optimization_result.best_params["growth_mu"]
        assert "fitness" in data["metadata"]

    def test_list_presets_function(self, temp_presets_dir, sample_params):
        """Test list_presets convenience function."""
        PresetManager.save_preset(sample_params, "list1")
        PresetManager.save_preset(sample_params, "list2")

        presets = list_presets()
        assert "list1" in presets
        assert "list2" in presets

    def test_get_preset_path_function(self, temp_presets_dir):
        """Test get_preset_path convenience function."""
        path = get_preset_path("test")
        assert "test" in str(path)
        assert path.suffix == ".yaml"

    def test_preset_exists_function(self, temp_presets_dir, sample_params):
        """Test preset_exists convenience function."""
        assert not preset_exists("exists_test")
        PresetManager.save_preset(sample_params, "exists_test")
        assert preset_exists("exists_test")

    def test_delete_preset_function(self, temp_presets_dir, sample_params):
        """Test delete_preset convenience function."""
        PresetManager.save_preset(sample_params, "delete_func_test")
        assert preset_exists("delete_func_test")

        result = delete_preset("delete_func_test")
        assert result is True
        assert not preset_exists("delete_func_test")


# =============================================================================
# Integration Tests
# =============================================================================


class TestPresetIntegration:
    """Integration tests with other modules."""

    def test_optimization_result_save_as_preset(self, temp_presets_dir, mock_optimization_result):
        """Test OptimizationResult.save_as_preset method."""
        path = mock_optimization_result.save_as_preset("result_preset")

        assert path.exists()
        config = get_optimized_config("result_preset")
        assert config.growth_mu == mock_optimization_result.best_params["growth_mu"]

    def test_optimization_result_to_field_config(self, mock_optimization_result):
        """Test OptimizationResult.to_field_config method."""
        config = mock_optimization_result.to_field_config()

        assert isinstance(config, FieldConfig)
        assert config.growth_mu == mock_optimization_result.best_params["growth_mu"]

    def test_optimization_result_to_field_config_with_overrides(self, mock_optimization_result):
        """Test to_field_config with overrides."""
        config = mock_optimization_result.to_field_config(width=2048, height=1024)

        assert config.width == 2048
        assert config.height == 1024
        assert config.growth_mu == mock_optimization_result.best_params["growth_mu"]

    def test_preset_with_lenia_field(self, temp_presets_dir, sample_params):
        """Test using preset with LeniaField."""
        from lenia_field import LeniaField

        PresetManager.save_preset(sample_params, "lenia_test")
        config = get_optimized_config("lenia_test")

        # Should be able to create LeniaField with config
        field = LeniaField(config)
        assert field.config.growth_mu == sample_params["growth_mu"]


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_save_with_metadata(self, temp_presets_dir, sample_params):
        """Test saving preset with custom metadata."""
        metadata = {"version": "1.0", "author": "test"}
        PresetManager.save_preset(sample_params, "meta_test", metadata=metadata)

        data = PresetManager.load_preset("meta_test")
        assert data["metadata"]["version"] == "1.0"
        assert data["metadata"]["author"] == "test"

    def test_load_old_format(self, temp_presets_dir):
        """Test loading preset in old format (just params, no metadata wrapper)."""
        import yaml

        old_format = {"growth_mu": 0.2, "decay": 0.05}
        path = temp_presets_dir / "old_format.yaml"
        with open(path, "w") as f:
            yaml.dump(old_format, f)

        data = PresetManager.load_preset("old_format")
        assert data["params"]["growth_mu"] == 0.2
        assert "metadata" in data

    def test_optimizable_params_list(self):
        """Test that OPTIMIZABLE_PARAMS contains expected params."""
        assert "growth_mu" in OPTIMIZABLE_PARAMS
        assert "decay" in OPTIMIZABLE_PARAMS
        assert "injection_power" in OPTIMIZABLE_PARAMS
        assert "r_max" in OPTIMIZABLE_PARAMS
