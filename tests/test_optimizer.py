"""Tests for optimizer.py module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from lenia_field.optimizer import (
    EvaluationMetrics,
    LeniaOptimizer,
    OptimizationResult,
    OptimizationTarget,
    ParameterSpec,
    quick_optimize,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_trajectories():
    """Simple static positions for quick tests."""
    np.random.seed(42)
    n_agents = 5
    positions = np.random.rand(n_agents, 2) * 200 + 28
    return positions.astype(np.float32)


@pytest.fixture
def dynamic_trajectories():
    """Dynamic trajectory data (random walk)."""
    np.random.seed(42)
    n_frames = 50
    n_agents = 5
    trajectories = np.zeros((n_frames, n_agents, 2), dtype=np.float32)
    trajectories[0] = np.random.rand(n_agents, 2) * 200 + 28

    for i in range(1, n_frames):
        step = np.random.randn(n_agents, 2) * 2
        trajectories[i] = np.clip(trajectories[i - 1] + step, 10, 246)

    return trajectories


@pytest.fixture
def small_optimizer():
    """Small optimizer for quick tests."""
    return LeniaOptimizer(
        width=64,
        height=64,
        suzuki_style=True,
        verbose=False,
    )


@pytest.fixture
def classic_optimizer():
    """Optimizer in classic (non-Suzuki) mode."""
    return LeniaOptimizer(
        width=64,
        height=64,
        suzuki_style=False,
        verbose=False,
    )


# =============================================================================
# ParameterSpec Tests
# =============================================================================


class TestParameterSpec:
    """Tests for ParameterSpec dataclass."""

    def test_creation(self):
        spec = ParameterSpec(
            name="test_param",
            bounds=(0.0, 1.0),
            default=0.5,
            description="A test parameter",
        )
        assert spec.name == "test_param"
        assert spec.bounds == (0.0, 1.0)
        assert spec.default == 0.5
        assert spec.description == "A test parameter"
        assert spec.log_scale is False

    def test_log_scale(self):
        spec = ParameterSpec(
            name="log_param",
            bounds=(0.001, 1.0),
            default=0.1,
            log_scale=True,
        )
        assert spec.log_scale is True


# =============================================================================
# OptimizationTarget Tests
# =============================================================================


class TestOptimizationTarget:
    """Tests for OptimizationTarget dataclass."""

    def test_default_values(self):
        target = OptimizationTarget()
        assert target.field_mean_target == 0.15
        assert target.field_mean_tolerance == 0.1
        assert target.field_std_min == 0.05
        assert target.field_std_max == 0.3
        assert target.temporal_change_min == 0.001
        assert target.temporal_change_max == 0.1

    def test_custom_values(self):
        target = OptimizationTarget(
            field_mean_target=0.2,
            field_std_min=0.1,
        )
        assert target.field_mean_target == 0.2
        assert target.field_std_min == 0.1


# =============================================================================
# EvaluationMetrics Tests
# =============================================================================


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_creation(self):
        metrics = EvaluationMetrics(
            fitness=10.0,
            field_mean=0.15,
            field_std=0.1,
            field_max=0.5,
            temporal_change=0.01,
            field_mean_stability=0.02,
        )
        assert metrics.fitness == 10.0
        assert metrics.field_mean == 0.15
        assert metrics.resource_mean is None

    def test_with_resource(self):
        metrics = EvaluationMetrics(
            fitness=10.0,
            field_mean=0.15,
            field_std=0.1,
            field_max=0.5,
            temporal_change=0.01,
            field_mean_stability=0.02,
            resource_mean=0.8,
        )
        assert metrics.resource_mean == 0.8

    def test_to_dict(self):
        metrics = EvaluationMetrics(
            fitness=10.0,
            field_mean=0.15,
            field_std=0.1,
            field_max=0.5,
            temporal_change=0.01,
            field_mean_stability=0.02,
        )
        d = metrics.to_dict()
        assert "fitness" in d
        assert "field_mean" in d
        assert d["fitness"] == 10.0
        assert "resource_mean" not in d

    def test_to_dict_with_resource(self):
        metrics = EvaluationMetrics(
            fitness=10.0,
            field_mean=0.15,
            field_std=0.1,
            field_max=0.5,
            temporal_change=0.01,
            field_mean_stability=0.02,
            resource_mean=0.8,
        )
        d = metrics.to_dict()
        assert "resource_mean" in d
        assert d["resource_mean"] == 0.8


# =============================================================================
# LeniaOptimizer Tests
# =============================================================================


class TestLeniaOptimizerInit:
    """Tests for LeniaOptimizer initialization."""

    def test_default_init(self):
        optimizer = LeniaOptimizer(verbose=False)
        assert optimizer.width == 512
        assert optimizer.height == 512
        assert optimizer.suzuki_style is True
        assert len(optimizer.param_names) > 0

    def test_custom_dimensions(self):
        optimizer = LeniaOptimizer(width=256, height=128, verbose=False)
        assert optimizer.width == 256
        assert optimizer.height == 128

    def test_suzuki_mode(self):
        opt_suzuki = LeniaOptimizer(suzuki_style=True, verbose=False)
        opt_classic = LeniaOptimizer(suzuki_style=False, verbose=False)

        # Suzuki mode should have more parameters
        assert len(opt_suzuki.param_names) > len(opt_classic.param_names)
        assert "r_max" in opt_suzuki.param_names
        assert "r_max" not in opt_classic.param_names

    def test_custom_target(self):
        target = OptimizationTarget(field_mean_target=0.3)
        optimizer = LeniaOptimizer(target=target, verbose=False)
        assert optimizer.target.field_mean_target == 0.3

    def test_custom_parameter_specs(self):
        custom_spec = ParameterSpec(
            name="custom_param",
            bounds=(0.0, 10.0),
            default=5.0,
        )
        optimizer = LeniaOptimizer(
            parameter_specs={"custom_param": custom_spec},
            suzuki_style=False,
            verbose=False,
        )
        assert "custom_param" in optimizer.param_names
        assert optimizer.parameter_specs["custom_param"].default == 5.0


class TestLeniaOptimizerHelpers:
    """Tests for LeniaOptimizer helper methods."""

    def test_get_bounds(self, small_optimizer):
        lower, upper = small_optimizer._get_bounds()
        assert len(lower) == len(small_optimizer.param_names)
        assert len(upper) == len(small_optimizer.param_names)
        assert all(l < u for l, u in zip(lower, upper))

    def test_params_to_vector(self, small_optimizer):
        params = {name: 0.5 for name in small_optimizer.param_names}
        vector = small_optimizer._params_to_vector(params)
        assert isinstance(vector, np.ndarray)
        assert len(vector) == len(small_optimizer.param_names)

    def test_vector_to_params(self, small_optimizer):
        vector = np.ones(len(small_optimizer.param_names)) * 0.5
        params = small_optimizer._vector_to_params(vector)
        assert isinstance(params, dict)
        assert all(name in params for name in small_optimizer.param_names)

    def test_random_params(self, small_optimizer):
        lower, upper = small_optimizer._get_bounds()
        params = small_optimizer._random_params()

        for i, name in enumerate(small_optimizer.param_names):
            assert lower[i] <= params[name] <= upper[i]

    def test_clip_params(self, small_optimizer):
        # Create params outside bounds
        params = {name: 1000.0 for name in small_optimizer.param_names}
        clipped = small_optimizer._clip_params(params)

        lower, upper = small_optimizer._get_bounds()
        for i, name in enumerate(small_optimizer.param_names):
            assert clipped[name] <= upper[i]


class TestLeniaOptimizerEvaluate:
    """Tests for LeniaOptimizer.evaluate()."""

    def test_evaluate_static_positions(self, small_optimizer, simple_trajectories):
        params = small_optimizer._random_params()
        metrics = small_optimizer.evaluate(
            params,
            simple_trajectories,
            n_warmup=5,
            n_eval=10,
        )

        assert isinstance(metrics, EvaluationMetrics)
        assert isinstance(metrics.fitness, float)
        assert 0.0 <= metrics.field_mean <= 1.0
        assert metrics.field_std >= 0.0

    def test_evaluate_dynamic_positions(self, small_optimizer, dynamic_trajectories):
        params = small_optimizer._random_params()
        metrics = small_optimizer.evaluate(
            params,
            dynamic_trajectories,
            n_warmup=5,
            n_eval=10,
        )

        assert isinstance(metrics, EvaluationMetrics)
        assert isinstance(metrics.fitness, float)

    def test_evaluate_suzuki_mode(self, small_optimizer, simple_trajectories):
        params = small_optimizer._random_params()
        metrics = small_optimizer.evaluate(
            params,
            simple_trajectories,
            n_warmup=5,
            n_eval=10,
        )

        # Suzuki mode should have resource_mean
        assert metrics.resource_mean is not None

    def test_evaluate_classic_mode(self, classic_optimizer, simple_trajectories):
        params = classic_optimizer._random_params()
        metrics = classic_optimizer.evaluate(
            params,
            simple_trajectories,
            n_warmup=5,
            n_eval=10,
        )

        # Classic mode should not have resource_mean
        assert metrics.resource_mean is None


class TestLeniaOptimizerOptimize:
    """Tests for LeniaOptimizer.optimize() - evolutionary strategy."""

    def test_optimize_basic(self, small_optimizer, simple_trajectories):
        result = small_optimizer.optimize(
            simple_trajectories,
            n_iterations=3,
            population_size=4,
            elite_size=2,
            n_warmup=3,
            n_eval=5,
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert len(result.best_params) > 0
        assert result.best_fitness > float("-inf")
        assert len(result.history) > 0

    def test_optimize_returns_valid_params(self, small_optimizer, simple_trajectories):
        result = small_optimizer.optimize(
            simple_trajectories,
            n_iterations=2,
            population_size=3,
            n_warmup=2,
            n_eval=3,
        )

        lower, upper = small_optimizer._get_bounds()
        for i, name in enumerate(small_optimizer.param_names):
            assert lower[i] <= result.best_params[name] <= upper[i]

    def test_optimize_history_structure(self, small_optimizer, simple_trajectories):
        result = small_optimizer.optimize(
            simple_trajectories,
            n_iterations=3,
            population_size=3,
            n_warmup=2,
            n_eval=3,
        )

        for entry in result.history:
            assert "iteration" in entry
            assert "best_fitness" in entry
            assert "best_params" in entry
            assert "best_metrics" in entry


class TestLeniaOptimizerGridSearch:
    """Tests for LeniaOptimizer.grid_search()."""

    def test_grid_search(self, small_optimizer, simple_trajectories):
        values = np.linspace(0.1, 0.2, 3)
        result = small_optimizer.grid_search(
            simple_trajectories,
            "growth_mu",
            values,
            n_warmup=2,
            n_eval=3,
        )

        assert "values" in result
        assert "fitness" in result
        assert "metrics" in result
        assert len(result["fitness"]) == 3

    def test_grid_search_invalid_param(self, small_optimizer, simple_trajectories):
        values = np.linspace(0.1, 0.2, 3)
        with pytest.raises(ValueError, match="Unknown parameter"):
            small_optimizer.grid_search(
                simple_trajectories,
                "nonexistent_param",
                values,
            )


class TestLeniaOptimizerSensitivityAnalysis:
    """Tests for LeniaOptimizer.sensitivity_analysis()."""

    def test_sensitivity_analysis(self, classic_optimizer, simple_trajectories):
        # Use classic mode with fewer parameters for faster test
        result = classic_optimizer.sensitivity_analysis(
            simple_trajectories,
            n_points=3,
            n_warmup=2,
            n_eval=3,
        )

        assert isinstance(result, dict)
        assert len(result) == len(classic_optimizer.param_names)

        for name in classic_optimizer.param_names:
            assert name in result
            assert "values" in result[name]
            assert "fitness" in result[name]


# =============================================================================
# OptimizationResult Tests
# =============================================================================


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        return OptimizationResult(
            best_params={"growth_mu": 0.15, "decay": 0.02},
            best_fitness=12.5,
            best_metrics={"field_mean": 0.15, "field_std": 0.1},
            history=[
                {
                    "iteration": 0,
                    "best_fitness": 10.0,
                    "best_params": {"growth_mu": 0.1, "decay": 0.01},
                    "best_metrics": {"field_mean": 0.1},
                },
                {
                    "iteration": 1,
                    "best_fitness": 12.5,
                    "best_params": {"growth_mu": 0.15, "decay": 0.02},
                    "best_metrics": {"field_mean": 0.15},
                },
            ],
            parameter_specs={
                "growth_mu": {"bounds": (0.05, 0.3), "default": 0.15},
                "decay": {"bounds": (0.005, 0.1), "default": 0.02},
            },
            target={"field_mean_target": 0.15},
            config={"width": 512, "height": 512},
            elapsed_time=10.5,
        )

    def test_save_load_roundtrip(self, sample_result):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            sample_result.save(path)
            loaded = OptimizationResult.load(path)

            assert loaded.best_fitness == sample_result.best_fitness
            assert loaded.best_params == sample_result.best_params
            assert len(loaded.history) == len(sample_result.history)
        finally:
            path.unlink()

    def test_summary(self, sample_result):
        summary = sample_result.summary()
        assert "OPTIMIZATION RESULTS" in summary
        assert "Best fitness: 12.5" in summary
        assert "growth_mu" in summary

    def test_save_handles_numpy_types(self, sample_result):
        # Add numpy types to the result
        sample_result.best_params["numpy_float"] = np.float64(1.5)
        sample_result.best_fitness = np.float32(12.5)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            sample_result.save(path)

            # Verify it can be loaded as valid JSON
            with open(path) as f:
                data = json.load(f)
            assert data["best_fitness"] == 12.5
        finally:
            path.unlink()


# =============================================================================
# Quick Optimize Function Tests
# =============================================================================


class TestQuickOptimize:
    """Tests for quick_optimize convenience function."""

    def test_quick_optimize(self, simple_trajectories):
        result = quick_optimize(
            simple_trajectories,
            width=64,
            height=64,
            n_iterations=2,
            verbose=False,
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert result.best_fitness > float("-inf")


# =============================================================================
# Fitness Computation Tests
# =============================================================================


class TestFitnessComputation:
    """Tests for fitness computation logic."""

    def test_good_metrics_give_positive_fitness(self, small_optimizer):
        metrics = EvaluationMetrics(
            fitness=0.0,  # Will be recomputed
            field_mean=0.15,  # Close to target
            field_std=0.15,  # In range
            field_max=0.5,  # Not saturated
            temporal_change=0.05,  # In range
            field_mean_stability=0.01,  # Stable
            resource_mean=0.5,  # In range
        )

        fitness = small_optimizer._compute_fitness(metrics)
        assert fitness > 0

    def test_dead_field_penalized(self, small_optimizer):
        metrics = EvaluationMetrics(
            fitness=0.0,
            field_mean=0.001,  # Dead field
            field_std=0.0,
            field_max=0.01,
            temporal_change=0.0,
            field_mean_stability=0.0,
        )

        fitness = small_optimizer._compute_fitness(metrics)
        assert fitness < 0

    def test_saturated_field_penalized(self, small_optimizer):
        metrics = EvaluationMetrics(
            fitness=0.0,
            field_mean=0.9,
            field_std=0.1,
            field_max=1.0,  # Saturated
            temporal_change=0.05,
            field_mean_stability=0.01,
        )

        fitness = small_optimizer._compute_fitness(metrics)
        assert fitness < 10  # Should lose saturation penalty

    def test_too_static_penalized(self, small_optimizer):
        metrics = EvaluationMetrics(
            fitness=0.0,
            field_mean=0.15,
            field_std=0.15,
            field_max=0.5,
            temporal_change=0.0001,  # Too static
            field_mean_stability=0.01,
        )

        fitness = small_optimizer._compute_fitness(metrics)
        # Compare with good temporal change
        good_metrics = EvaluationMetrics(
            fitness=0.0,
            field_mean=0.15,
            field_std=0.15,
            field_max=0.5,
            temporal_change=0.05,  # Good
            field_mean_stability=0.01,
        )
        good_fitness = small_optimizer._compute_fitness(good_metrics)
        assert fitness < good_fitness


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_trajectories(self, small_optimizer):
        empty = np.zeros((0, 2), dtype=np.float32)
        params = small_optimizer._random_params()

        # Should not crash
        metrics = small_optimizer.evaluate(
            params,
            empty,
            n_warmup=2,
            n_eval=3,
        )
        assert isinstance(metrics, EvaluationMetrics)

    def test_single_agent(self, small_optimizer):
        single = np.array([[32.0, 32.0]], dtype=np.float32)
        params = small_optimizer._random_params()

        metrics = small_optimizer.evaluate(
            params,
            single,
            n_warmup=2,
            n_eval=3,
        )
        assert isinstance(metrics, EvaluationMetrics)

    def test_trajectory_cycling(self, small_optimizer):
        """Test that short trajectories are cycled correctly."""
        short_traj = np.random.rand(5, 3, 2).astype(np.float32) * 50 + 10
        params = small_optimizer._random_params()

        # Request more steps than trajectory length
        metrics = small_optimizer.evaluate(
            params,
            short_traj,
            n_warmup=10,  # More than 5 frames
            n_eval=10,
        )
        assert isinstance(metrics, EvaluationMetrics)
