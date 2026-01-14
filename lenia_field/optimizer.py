"""
Parameter optimizer for Lenia field simulation.

Finds optimal parameters to maintain "unstable equilibrium" - dynamic patterns
that neither collapse (field -> 0) nor explode (field -> 1).

Features:
- Evolutionary optimization
- Scipy optimization (L-BFGS-B, Powell, etc.)
- Grid search for sensitivity analysis
- Parameter space visualization (requires matplotlib: pip install lenia_field[viz])
- JSON results storage and loading

Usage:
    from lenia_field.optimizer import LeniaOptimizer, OptimizationResult

    # Load trajectory data (N_frames x N_agents x 2)
    trajectories = np.load("my_trajectories.npy")

    # Optimize
    optimizer = LeniaOptimizer(width=512, height=512)
    result = optimizer.optimize(trajectories, n_iterations=100)

    # Save results
    result.save("optimization_result.json")

    # Plot results (requires matplotlib)
    result.plot_convergence()
    result.plot_parameter_importance()
"""

import json
import time
import warnings
from dataclasses import dataclass, field, asdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .core import FieldConfig, LeniaField


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ParameterSpec:
    """Specification for a single optimizable parameter."""

    name: str
    bounds: Tuple[float, float]
    default: float
    description: str = ""
    log_scale: bool = False  # Whether to sample in log space


@dataclass
class OptimizationTarget:
    """Target metrics for optimization."""

    # Field statistics
    field_mean_target: float = 0.15
    field_mean_tolerance: float = 0.1
    field_std_min: float = 0.05
    field_std_max: float = 0.3

    # Temporal dynamics
    temporal_change_min: float = 0.001
    temporal_change_max: float = 0.1

    # Resource balance (Suzuki mode)
    resource_mean_min: float = 0.2
    resource_mean_max: float = 0.9


@dataclass
class EvaluationMetrics:
    """Metrics from a single parameter evaluation."""

    fitness: float
    field_mean: float
    field_std: float
    field_max: float
    temporal_change: float
    field_mean_stability: float
    resource_mean: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        d = {
            "fitness": self.fitness,
            "field_mean": self.field_mean,
            "field_std": self.field_std,
            "field_max": self.field_max,
            "temporal_change": self.temporal_change,
            "field_mean_stability": self.field_mean_stability,
        }
        if self.resource_mean is not None:
            d["resource_mean"] = self.resource_mean
        return d


@dataclass
class OptimizationResult:
    """Complete optimization results with serialization and plotting."""

    best_params: Dict[str, float]
    best_fitness: float
    best_metrics: Dict[str, float]
    history: List[Dict[str, Any]]
    parameter_specs: Dict[str, Dict[str, Any]]
    target: Dict[str, float]
    config: Dict[str, Any]
    elapsed_time: float = 0.0

    def save(self, path: Union[str, Path]) -> None:
        """Save results to JSON file."""
        path = Path(path)

        data = {
            "best_params": self.best_params,
            "best_fitness": self.best_fitness,
            "best_metrics": self.best_metrics,
            "history": self.history,
            "parameter_specs": self.parameter_specs,
            "target": self.target,
            "config": self.config,
            "elapsed_time": self.elapsed_time,
        }

        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=convert)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OptimizationResult":
        """Load results from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def summary(self) -> str:
        """Return a summary string of the optimization results."""
        lines = [
            "=" * 60,
            "OPTIMIZATION RESULTS",
            "=" * 60,
            f"Best fitness: {self.best_fitness:.4f}",
            f"Elapsed time: {self.elapsed_time:.1f}s",
            f"Iterations: {len(self.history)}",
            "",
            "Best Parameters:",
        ]
        for name, value in self.best_params.items():
            lines.append(f"  {name}: {value:.6f}")
        lines.append("")
        lines.append("Best Metrics:")
        for name, value in self.best_metrics.items():
            lines.append(f"  {name}: {value:.6f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def save_as_preset(self, name: str = "optimized", format: str = "yaml") -> Path:
        """
        Save the best parameters as a preset for later use.

        The preset can be loaded with get_optimized_config(name).

        Args:
            name: Preset name (default: "optimized")
            format: File format ('yaml' or 'json')

        Returns:
            Path to saved preset file

        Example:
            result = optimizer.optimize(trajectories)
            result.save_as_preset("my_optimized")

            # Later:
            config = get_optimized_config("my_optimized")
        """
        from .presets import save_optimized_preset

        return save_optimized_preset(self, name=name, format=format)

    def to_field_config(self, **overrides) -> "FieldConfig":
        """
        Create a FieldConfig from the optimized parameters.

        This allows direct use of optimization results with LeniaField,
        Simulation, or any other component.

        Args:
            **overrides: Override specific parameters (e.g., width=1024)

        Returns:
            FieldConfig with optimized parameters

        Example:
            result = optimizer.optimize(trajectories)
            config = result.to_field_config(width=1024, height=1024)
            field = LeniaField(config)
        """
        params = self.best_params.copy()
        params.update(overrides)
        return FieldConfig(**params)

    # -------------------------------------------------------------------------
    # Plotting methods (require matplotlib)
    # -------------------------------------------------------------------------

    def _check_matplotlib(self):
        """Check if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt

            return plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install lenia_field[viz]"
            )

    def plot_convergence(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ):
        """
        Plot fitness convergence over iterations.

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        plt = self._check_matplotlib()

        iterations = [h["iteration"] for h in self.history]
        best_fitness = [h["best_fitness"] for h in self.history]
        mean_fitness = [h.get("mean_fitness", h["best_fitness"]) for h in self.history]

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(iterations, best_fitness, "b-", linewidth=2, label="Best Fitness")
        ax.plot(iterations, mean_fitness, "g--", alpha=0.7, label="Mean Fitness")
        ax.fill_between(iterations, mean_fitness, best_fitness, alpha=0.3)

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Fitness", fontsize=12)
        ax.set_title("Optimization Convergence", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax

    def plot_parameter_history(
        self,
        params: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ):
        """
        Plot parameter values over iterations.

        Args:
            params: List of parameter names to plot (None = all)
            figsize: Figure size
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        plt = self._check_matplotlib()

        if params is None:
            params = list(self.best_params.keys())

        n_params = len(params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_params == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, param in enumerate(params):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            iterations = [h["iteration"] for h in self.history]
            values = [h["best_params"].get(param, np.nan) for h in self.history]

            ax.plot(iterations, values, "b-", linewidth=1.5)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(param)
            ax.set_title(param, fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add bounds if available
            if param in self.parameter_specs:
                bounds = self.parameter_specs[param]["bounds"]
                ax.axhline(bounds[0], color="r", linestyle="--", alpha=0.5)
                ax.axhline(bounds[1], color="r", linestyle="--", alpha=0.5)

        # Hide empty subplots
        for idx in range(n_params, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

        return fig, axes

    def plot_parameter_importance(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ):
        """
        Plot parameter importance based on correlation with fitness.

        Computes Spearman correlation between each parameter and fitness
        across all evaluated points in history.

        Args:
            figsize: Figure size
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        plt = self._check_matplotlib()
        from scipy import stats

        # Collect all parameter values and fitness scores
        param_names = list(self.best_params.keys())
        all_params = {name: [] for name in param_names}
        all_fitness = []

        for h in self.history:
            all_fitness.append(h["best_fitness"])
            for name in param_names:
                all_params[name].append(h["best_params"].get(name, np.nan))

        # Compute correlations
        correlations = {}
        for name in param_names:
            values = np.array(all_params[name])
            fitness = np.array(all_fitness)
            valid = ~(np.isnan(values) | np.isnan(fitness))
            if valid.sum() > 2:
                corr, _ = stats.spearmanr(values[valid], fitness[valid])
                correlations[name] = abs(corr) if not np.isnan(corr) else 0
            else:
                correlations[name] = 0

        # Sort by importance
        sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        names = [p[0] for p in sorted_params]
        values = [p[1] for p in sorted_params]

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        bars = ax.barh(range(len(names)), values, color=colors)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Absolute Correlation with Fitness", fontsize=12)
        ax.set_title("Parameter Importance", fontsize=14)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax

    def plot_metrics_evolution(
        self,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ):
        """
        Plot how metrics evolved during optimization.

        Args:
            metrics: List of metric names to plot (None = all)
            figsize: Figure size
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        plt = self._check_matplotlib()

        # Get available metrics
        if self.history and "best_metrics" in self.history[0]:
            available = list(self.history[0]["best_metrics"].keys())
        else:
            available = list(self.best_metrics.keys())

        if metrics is None:
            metrics = available

        fig, ax = plt.subplots(figsize=figsize)

        iterations = [h["iteration"] for h in self.history]
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))

        for metric, color in zip(metrics, colors):
            values = [h.get("best_metrics", {}).get(metric, np.nan) for h in self.history]
            ax.plot(iterations, values, label=metric, color=color, linewidth=1.5)

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_title("Metrics Evolution", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax

    def plot_parallel_coordinates(
        self,
        top_n: int = 20,
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ):
        """
        Plot parallel coordinates of top parameter combinations.

        Each line represents a parameter set, with color indicating fitness.

        Args:
            top_n: Number of top solutions to plot
            figsize: Figure size
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        plt = self._check_matplotlib()
        from matplotlib.colors import Normalize

        param_names = list(self.best_params.keys())
        n_params = len(param_names)

        # Collect unique parameter sets with their fitness
        seen = set()
        unique_solutions = []
        for h in self.history:
            key = tuple(sorted(h["best_params"].items()))
            if key not in seen:
                seen.add(key)
                unique_solutions.append((h["best_fitness"], h["best_params"]))

        # Sort by fitness and take top_n
        unique_solutions.sort(key=lambda x: x[0], reverse=True)
        unique_solutions = unique_solutions[:top_n]

        if not unique_solutions:
            print("No solutions to plot")
            return None, None

        fig, ax = plt.subplots(figsize=figsize)

        # Normalize parameters to [0, 1] for visualization
        bounds = {}
        for name in param_names:
            if name in self.parameter_specs:
                bounds[name] = self.parameter_specs[name]["bounds"]
            else:
                values = [s[1].get(name, 0) for s in unique_solutions]
                bounds[name] = (min(values), max(values))

        # Normalize fitness for coloring
        fitness_values = [s[0] for s in unique_solutions]
        norm = Normalize(vmin=min(fitness_values), vmax=max(fitness_values))
        cmap = plt.cm.RdYlGn

        # Plot each solution
        x = np.arange(n_params)
        for fitness, params in unique_solutions:
            y = []
            for name in param_names:
                val = params.get(name, 0)
                lo, hi = bounds[name]
                if hi > lo:
                    y.append((val - lo) / (hi - lo))
                else:
                    y.append(0.5)

            color = cmap(norm(fitness))
            ax.plot(x, y, "-o", color=color, alpha=0.7, linewidth=1.5, markersize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.set_ylabel("Normalized Value", fontsize=12)
        ax.set_title(f"Top {len(unique_solutions)} Parameter Combinations", fontsize=14)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Fitness", fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax

    def plot_all(
        self,
        save_dir: Optional[Union[str, Path]] = None,
        show: bool = True,
    ):
        """
        Generate all available plots.

        Args:
            save_dir: Optional directory to save plots
            show: Whether to display plots
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        plots = []

        try:
            fig, ax = self.plot_convergence(
                save_path=save_dir / "convergence.png" if save_dir else None,
                show=show,
            )
            plots.append(("convergence", fig))
        except Exception as e:
            warnings.warn(f"Could not plot convergence: {e}")

        try:
            fig, ax = self.plot_parameter_history(
                save_path=save_dir / "parameter_history.png" if save_dir else None,
                show=show,
            )
            plots.append(("parameter_history", fig))
        except Exception as e:
            warnings.warn(f"Could not plot parameter history: {e}")

        try:
            fig, ax = self.plot_parameter_importance(
                save_path=save_dir / "parameter_importance.png" if save_dir else None,
                show=show,
            )
            plots.append(("parameter_importance", fig))
        except Exception as e:
            warnings.warn(f"Could not plot parameter importance: {e}")

        try:
            fig, ax = self.plot_metrics_evolution(
                save_path=save_dir / "metrics_evolution.png" if save_dir else None,
                show=show,
            )
            plots.append(("metrics_evolution", fig))
        except Exception as e:
            warnings.warn(f"Could not plot metrics evolution: {e}")

        try:
            fig, ax = self.plot_parallel_coordinates(
                save_path=save_dir / "parallel_coordinates.png" if save_dir else None,
                show=show,
            )
            plots.append(("parallel_coordinates", fig))
        except Exception as e:
            warnings.warn(f"Could not plot parallel coordinates: {e}")

        return plots


# =============================================================================
# Optimizer Class
# =============================================================================


class LeniaOptimizer:
    """
    Optimizer to find parameters that maintain unstable equilibrium.

    The optimizer evaluates parameter sets by running simulations and
    measuring how well they maintain dynamic, balanced patterns.
    """

    # Default parameter specifications
    DEFAULT_PARAMS = {
        # Growth function
        "growth_mu": ParameterSpec(
            "growth_mu",
            (0.05, 0.3),
            0.15,
            "Center of Lenia growth function",
        ),
        "growth_sigma": ParameterSpec(
            "growth_sigma",
            (0.005, 0.05),
            0.015,
            "Width of Lenia growth function",
        ),
        "growth_amplitude": ParameterSpec(
            "growth_amplitude",
            (0.5, 2.0),
            1.0,
            "Amplitude of growth response",
        ),
        # Dynamics
        "dt": ParameterSpec(
            "dt",
            (0.05, 0.2),
            0.1,
            "Time step size",
        ),
        "decay": ParameterSpec(
            "decay",
            (0.005, 0.1),
            0.02,
            "Passive decay rate",
        ),
        # Injection
        "injection_power": ParameterSpec(
            "injection_power",
            (0.05, 2.0),
            0.1,
            "Injection intensity per agent",
        ),
        "injection_radius": ParameterSpec(
            "injection_radius",
            (2.0, 15.0),
            5.0,
            "Radius of injection spot",
        ),
    }

    SUZUKI_PARAMS = {
        "r_max": ParameterSpec(
            "r_max",
            (0.5, 3.0),
            1.0,
            "Maximum resource per cell",
        ),
        "R_C": ParameterSpec(
            "R_C",
            (0.001, 0.02),
            0.005,
            "Resource consumption by morphogen",
        ),
        "R_G": ParameterSpec(
            "R_G",
            (0.001, 0.01),
            0.002,
            "Resource recovery rate",
        ),
        "resource_effect_rate": ParameterSpec(
            "resource_effect_rate",
            (0.1, 1.0),
            0.5,
            "Multiplier for agent resource effect",
        ),
    }

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        target: Optional[OptimizationTarget] = None,
        parameter_specs: Optional[Dict[str, ParameterSpec]] = None,
        suzuki_style: bool = True,
        seed: str = "default",
        verbose: bool = True,
    ):
        """
        Initialize the optimizer.

        Args:
            width: Field width
            height: Field height
            target: Target metrics for optimization
            parameter_specs: Custom parameter specifications (overrides defaults)
            suzuki_style: Whether to use Suzuki resource dynamics
            seed: Initialization seed for optimization:
                - "default": Use values from default.yaml as starting point
                - "random": Initialize randomly within parameter bounds
                - preset name (e.g., "optimized"): Use that preset's values
            verbose: Print progress information
        """
        self.width = width
        self.height = height
        self.target = target or OptimizationTarget()
        self.suzuki_style = suzuki_style
        self.seed = seed
        self.verbose = verbose

        # Build parameter specs
        self.parameter_specs = dict(self.DEFAULT_PARAMS)
        if suzuki_style:
            self.parameter_specs.update(self.SUZUKI_PARAMS)
        if parameter_specs:
            self.parameter_specs.update(parameter_specs)

        self.param_names = list(self.parameter_specs.keys())

        # Load seed parameters
        self._seed_params = self._load_seed_params()

        # Results tracking
        self._best_params: Optional[Dict[str, float]] = None
        self._best_fitness: float = float("-inf")
        self._best_metrics: Optional[Dict[str, float]] = None
        self._history: List[Dict[str, Any]] = []
        self._start_time: float = 0

    def _load_seed_params(self) -> Dict[str, float]:
        """Load seed parameters from the specified source."""
        if self.seed == "random":
            # Use parameter spec defaults as fallback
            return {name: spec.default for name, spec in self.parameter_specs.items()}

        if self.seed == "default":
            # Load from default.yaml
            try:
                from .config_manager import load_config
                config = load_config()
                field_config = config.get("field", {})
                params = {}
                for name in self.param_names:
                    if name in field_config:
                        params[name] = float(field_config[name])
                    else:
                        params[name] = self.parameter_specs[name].default
                return params
            except Exception:
                # Fall back to parameter spec defaults
                return {name: spec.default for name, spec in self.parameter_specs.items()}

        # Try to load from a preset
        try:
            from .presets import PresetManager
            preset_data = PresetManager.load_preset(self.seed)
            params = {}
            for name in self.param_names:
                if name in preset_data["params"]:
                    params[name] = float(preset_data["params"][name])
                else:
                    params[name] = self.parameter_specs[name].default
            return params
        except FileNotFoundError:
            if self.verbose:
                print(f"Warning: Seed preset '{self.seed}' not found, using defaults")
            return {name: spec.default for name, spec in self.parameter_specs.items()}

    def get_seed_params(self) -> Dict[str, float]:
        """Get the seed parameters used for initialization."""
        return self._seed_params.copy()

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get lower and upper bounds as arrays."""
        lower = np.array([self.parameter_specs[n].bounds[0] for n in self.param_names])
        upper = np.array([self.parameter_specs[n].bounds[1] for n in self.param_names])
        return lower, upper

    def _params_to_vector(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to optimization vector."""
        return np.array([params.get(n, self.parameter_specs[n].default) for n in self.param_names])

    def _vector_to_params(self, vector: np.ndarray) -> Dict[str, float]:
        """Convert optimization vector to parameter dict."""
        return {name: float(vector[i]) for i, name in enumerate(self.param_names)}

    def _random_params(self, perturbation: float = 1.0) -> Dict[str, float]:
        """
        Generate parameters by perturbing around seed values.

        Args:
            perturbation: How much to perturb (0.0 = seed values, 1.0 = full random)

        Returns:
            Dict of parameter values
        """
        lower, upper = self._get_bounds()
        param_range = upper - lower

        if self.seed == "random" or perturbation >= 1.0:
            # Fully random
            vector = np.random.uniform(lower, upper)
        else:
            # Perturb around seed values
            seed_vector = self._params_to_vector(self._seed_params)
            noise = np.random.randn(len(seed_vector)) * perturbation * param_range * 0.3
            vector = np.clip(seed_vector + noise, lower, upper)

        return self._vector_to_params(vector)

    def _init_population(self, population_size: int) -> List[Dict[str, float]]:
        """
        Initialize population for optimization.

        The first individual is always the seed parameters (unperturbed).
        Remaining individuals are perturbations around the seed.
        """
        population = []

        # First individual: exact seed parameters (clipped to bounds)
        population.append(self._clip_params(self._seed_params))

        # Rest: perturbations around seed
        for i in range(1, population_size):
            # Gradually increase perturbation for diversity
            perturbation = 0.3 + 0.7 * (i / population_size)
            population.append(self._random_params(perturbation))

        return population

    def _clip_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Clip parameters to bounds."""
        result = {}
        for name in self.param_names:
            spec = self.parameter_specs[name]
            val = params.get(name, spec.default)
            result[name] = float(np.clip(val, spec.bounds[0], spec.bounds[1]))
        return result

    def evaluate(
        self,
        params: Dict[str, float],
        trajectories: np.ndarray,
        n_warmup: int = 50,
        n_eval: int = 200,
    ) -> EvaluationMetrics:
        """
        Evaluate fitness of parameters on trajectory data.

        Args:
            params: Parameter dict to evaluate
            trajectories: Shape (n_frames, n_agents, 2) or (n_agents, 2)
            n_warmup: Number of warmup steps before evaluation
            n_eval: Number of evaluation steps

        Returns:
            EvaluationMetrics with fitness and all metrics
        """
        # Build config
        config = FieldConfig(
            width=self.width,
            height=self.height,
            suzuki_style=self.suzuki_style,
            skip_diffusion=True,
            **self._clip_params(params),
        )

        lenia = LeniaField(config)

        # Handle trajectory formats
        if trajectories.ndim == 2:
            positions_sequence = [trajectories] * (n_warmup + n_eval)
        else:
            n_frames = trajectories.shape[0]
            total_steps = n_warmup + n_eval
            indices = np.arange(total_steps) % n_frames
            positions_sequence = [trajectories[i] for i in indices]

        # Warmup
        for i in range(n_warmup):
            lenia.step(positions=positions_sequence[i])

        # Evaluation
        field_means = []
        field_stds = []
        field_maxs = []
        temporal_changes = []
        resource_means = []

        prev_field = lenia.get_field().copy()

        for i in range(n_eval):
            field = lenia.step(positions=positions_sequence[n_warmup + i])

            field_means.append(np.mean(field))
            field_stds.append(np.std(field))
            field_maxs.append(np.max(field))

            change = np.mean(np.abs(field - prev_field))
            temporal_changes.append(change)
            prev_field = field.copy()

            if self.suzuki_style:
                resource_means.append(np.mean(lenia.get_resource()))

        # Compute metrics
        metrics = EvaluationMetrics(
            fitness=0.0,  # Will be computed
            field_mean=float(np.mean(field_means)),
            field_std=float(np.mean(field_stds)),
            field_max=float(np.mean(field_maxs)),
            temporal_change=float(np.mean(temporal_changes)),
            field_mean_stability=float(np.std(field_means)),
            resource_mean=float(np.mean(resource_means)) if resource_means else None,
        )

        # Compute fitness
        metrics.fitness = self._compute_fitness(metrics)

        return metrics

    def _compute_fitness(self, metrics: EvaluationMetrics) -> float:
        """Compute fitness score from metrics. Higher is better."""
        t = self.target
        fitness = 0.0

        # Field mean - close to target
        mean_error = abs(metrics.field_mean - t.field_mean_target)
        if mean_error < t.field_mean_tolerance:
            fitness += 10.0 * (1.0 - mean_error / t.field_mean_tolerance)
        else:
            fitness -= 5.0 * mean_error

        # Field std - in acceptable range
        if t.field_std_min <= metrics.field_std <= t.field_std_max:
            mid = (t.field_std_min + t.field_std_max) / 2
            span = t.field_std_max - t.field_std_min
            fitness += 5.0 * (1.0 - abs(metrics.field_std - mid) / span)
        else:
            fitness -= 10.0

        # Temporal dynamics
        if t.temporal_change_min <= metrics.temporal_change <= t.temporal_change_max:
            fitness += 5.0
        elif metrics.temporal_change < t.temporal_change_min:
            fitness -= 10.0  # Too static
        else:
            fitness -= 5.0 * (metrics.temporal_change - t.temporal_change_max)

        # Stability
        fitness -= 2.0 * metrics.field_mean_stability

        # Resource balance (Suzuki mode)
        if metrics.resource_mean is not None:
            if t.resource_mean_min <= metrics.resource_mean <= t.resource_mean_max:
                fitness += 3.0
            else:
                fitness -= 5.0

        # Penalties
        if metrics.field_max > 0.99:
            fitness -= 5.0  # Saturation
        if metrics.field_mean < 0.01:
            fitness -= 20.0  # Dead field

        return float(fitness)

    def _update_best(
        self, params: Dict[str, float], metrics: EvaluationMetrics
    ) -> bool:
        """Update best solution if this one is better. Returns True if updated."""
        if metrics.fitness > self._best_fitness:
            self._best_fitness = metrics.fitness
            self._best_params = params.copy()
            self._best_metrics = metrics.to_dict()
            return True
        return False

    def _record_history(
        self,
        iteration: int,
        params: Dict[str, float],
        metrics: EvaluationMetrics,
        mean_fitness: Optional[float] = None,
    ):
        """Record iteration to history."""
        self._history.append(
            {
                "iteration": iteration,
                "best_fitness": self._best_fitness,
                "mean_fitness": mean_fitness or metrics.fitness,
                "best_params": self._best_params.copy() if self._best_params else params.copy(),
                "best_metrics": self._best_metrics.copy() if self._best_metrics else metrics.to_dict(),
            }
        )

    def optimize(
        self,
        trajectories: np.ndarray,
        n_iterations: int = 100,
        population_size: int = 20,
        elite_size: int = 4,
        mutation_rate: float = 0.3,
        n_warmup: int = 50,
        n_eval: int = 500,
        early_stop_patience: int = 20,
        min_improvement: float = 0.01,
    ) -> OptimizationResult:
        """
        Optimize parameters using evolutionary strategy.

        The optimizer minimizes a loss function (negative fitness) that measures
        how well parameters maintain dynamic, balanced patterns. Lower loss is better.

        Args:
            trajectories: Agent positions, shape (n_frames, n_agents, 2) or (n_agents, 2)
            n_iterations: Number of optimization iterations
            population_size: Number of parameter sets per iteration
            elite_size: Number of best sets to keep
            mutation_rate: Relative mutation strength
            n_warmup: Warmup steps per evaluation
            n_eval: Evaluation steps per evaluation
            early_stop_patience: Stop if no improvement for this many iterations
            min_improvement: Minimum fitness improvement to reset patience counter

        Returns:
            OptimizationResult with best parameters and history
        """
        self._start_time = time.time()
        self._history = []
        self._best_fitness = float("-inf")

        if self.verbose:
            print(f"Starting evolutionary optimization...")
            print(f"  Seed: {self.seed}")
            print(f"  Iterations: {n_iterations}, Population: {population_size}")
            print(f"  Parameters: {len(self.param_names)}")
            print(f"  Early stop patience: {early_stop_patience} iterations")

        # Initialize population around seed values
        population = self._init_population(population_size)
        lower, upper = self._get_bounds()
        param_range = upper - lower

        # Early stopping tracking
        iterations_without_improvement = 0
        previous_best = float("-inf")

        for iteration in range(n_iterations):
            # Evaluate population
            results = []
            for params in population:
                metrics = self.evaluate(params, trajectories, n_warmup, n_eval)
                results.append((metrics.fitness, params, metrics))
                self._update_best(params, metrics)

            # Sort by fitness (higher is better, so we maximize)
            results.sort(key=lambda x: x[0], reverse=True)
            mean_fitness = np.mean([r[0] for r in results])
            best_this_iter = results[0][0]

            # Record history
            self._record_history(
                iteration,
                results[0][1],
                results[0][2],
                mean_fitness,
            )

            # Report loss (negative fitness) for clarity
            loss = -best_this_iter
            if self.verbose and iteration % 5 == 0:
                m = results[0][2]
                print(
                    f"Iter {iteration:4d}: loss={loss:8.3f}, "
                    f"mean={m.field_mean:.3f}, std={m.field_std:.3f}, "
                    f"temporal={m.temporal_change:.4f}"
                )

            # Check for improvement (convergence-based early stopping)
            if best_this_iter > previous_best + min_improvement:
                iterations_without_improvement = 0
                previous_best = best_this_iter
            else:
                iterations_without_improvement += 1

            if iterations_without_improvement >= early_stop_patience:
                if self.verbose:
                    print(f"Converged at iteration {iteration} (no improvement for {early_stop_patience} iterations)")
                break

            # Selection and mutation
            elite = [r[1] for r in results[:elite_size]]
            new_population = elite.copy()

            while len(new_population) < population_size:
                parent = elite[np.random.randint(len(elite))]
                child_vec = self._params_to_vector(parent)
                mutation = np.random.randn(len(child_vec)) * mutation_rate * param_range
                child_vec = np.clip(child_vec + mutation, lower, upper)
                new_population.append(self._vector_to_params(child_vec))

            population = new_population

        elapsed = time.time() - self._start_time

        if self.verbose:
            final_loss = -self._best_fitness
            print(f"\nOptimization complete in {elapsed:.1f}s")
            print(f"Final loss: {final_loss:.3f} (fitness: {self._best_fitness:.3f})")

        return self._build_result(elapsed)

    def optimize_scipy(
        self,
        trajectories: np.ndarray,
        method: str = "L-BFGS-B",
        n_warmup: int = 50,
        n_eval: int = 200,
        maxiter: int = 100,
        n_restarts: int = 3,
    ) -> OptimizationResult:
        """
        Optimize using scipy.optimize.

        Args:
            trajectories: Agent positions
            method: Scipy method ('L-BFGS-B', 'Powell', 'Nelder-Mead')
            n_warmup: Warmup steps per evaluation
            n_eval: Evaluation steps per evaluation
            maxiter: Maximum iterations per restart
            n_restarts: Number of random restarts

        Returns:
            OptimizationResult
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError("scipy required. Install with: pip install scipy")

        self._start_time = time.time()
        self._history = []
        self._best_fitness = float("-inf")

        lower, upper = self._get_bounds()
        bounds = list(zip(lower, upper))

        if self.verbose:
            print(f"Starting scipy optimization ({method})...")
            print(f"  Restarts: {n_restarts}, Max iterations: {maxiter}")

        eval_count = [0]

        def objective(x):
            params = self._vector_to_params(x)
            metrics = self.evaluate(params, trajectories, n_warmup, n_eval)
            self._update_best(params, metrics)
            self._record_history(eval_count[0], params, metrics)
            eval_count[0] += 1
            return -metrics.fitness

        for restart in range(n_restarts):
            if restart == 0:
                x0 = (lower + upper) / 2
            else:
                x0 = np.random.uniform(lower, upper)

            if self.verbose:
                print(f"  Restart {restart + 1}/{n_restarts}...")

            result = minimize(
                objective,
                x0,
                method=method,
                bounds=bounds,
                options={"maxiter": maxiter, "disp": False},
            )

            if self.verbose:
                print(f"    Fitness: {-result.fun:.3f}")

        elapsed = time.time() - self._start_time

        if self.verbose:
            print(f"\nOptimization complete in {elapsed:.1f}s")
            print(f"Best fitness: {self._best_fitness:.3f}")

        return self._build_result(elapsed)

    def grid_search(
        self,
        trajectories: np.ndarray,
        param_name: str,
        values: np.ndarray,
        base_params: Optional[Dict[str, float]] = None,
        n_warmup: int = 50,
        n_eval: int = 200,
    ) -> Dict[str, Any]:
        """
        Perform grid search over a single parameter.

        Useful for sensitivity analysis and understanding parameter impact.

        Args:
            trajectories: Agent positions
            param_name: Name of parameter to vary
            values: Array of values to test
            base_params: Base parameter values (defaults used if None)
            n_warmup: Warmup steps
            n_eval: Evaluation steps

        Returns:
            Dict with 'values', 'fitness', 'metrics' arrays
        """
        if param_name not in self.param_names:
            raise ValueError(f"Unknown parameter: {param_name}")

        if base_params is None:
            base_params = {n: self.parameter_specs[n].default for n in self.param_names}

        if self.verbose:
            print(f"Grid search over {param_name}: {len(values)} values")

        results = {"values": values, "fitness": [], "metrics": []}

        for i, val in enumerate(values):
            params = base_params.copy()
            params[param_name] = float(val)
            metrics = self.evaluate(params, trajectories, n_warmup, n_eval)
            results["fitness"].append(metrics.fitness)
            results["metrics"].append(metrics.to_dict())

            if self.verbose and i % 5 == 0:
                print(f"  {param_name}={val:.4f}: fitness={metrics.fitness:.3f}")

        results["fitness"] = np.array(results["fitness"])
        return results

    def sensitivity_analysis(
        self,
        trajectories: np.ndarray,
        n_points: int = 10,
        base_params: Optional[Dict[str, float]] = None,
        n_warmup: int = 50,
        n_eval: int = 100,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform sensitivity analysis on all parameters.

        Varies each parameter while keeping others at base values.

        Args:
            trajectories: Agent positions
            n_points: Number of points per parameter
            base_params: Base parameter values
            n_warmup: Warmup steps
            n_eval: Evaluation steps

        Returns:
            Dict mapping parameter name to grid search results
        """
        if self.verbose:
            print(f"Sensitivity analysis: {len(self.param_names)} parameters x {n_points} points")

        results = {}
        for name in self.param_names:
            spec = self.parameter_specs[name]
            values = np.linspace(spec.bounds[0], spec.bounds[1], n_points)
            results[name] = self.grid_search(
                trajectories, name, values, base_params, n_warmup, n_eval
            )

        return results

    def _build_result(self, elapsed: float) -> OptimizationResult:
        """Build OptimizationResult from current state."""
        return OptimizationResult(
            best_params=self._best_params or {},
            best_fitness=self._best_fitness,
            best_metrics=self._best_metrics or {},
            history=self._history,
            parameter_specs={
                name: {"bounds": spec.bounds, "default": spec.default, "description": spec.description}
                for name, spec in self.parameter_specs.items()
            },
            target=asdict(self.target),
            config={
                "width": self.width,
                "height": self.height,
                "suzuki_style": self.suzuki_style,
            },
            elapsed_time=elapsed,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_optimize(
    trajectories: np.ndarray,
    width: int = 512,
    height: int = 512,
    n_iterations: int = 50,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Quick optimization with default settings.

    Args:
        trajectories: Agent positions, shape (n_frames, n_agents, 2) or (n_agents, 2)
        width: Field width
        height: Field height
        n_iterations: Number of iterations
        verbose: Print progress

    Returns:
        OptimizationResult
    """
    optimizer = LeniaOptimizer(width=width, height=height, verbose=verbose)
    return optimizer.optimize(trajectories, n_iterations=n_iterations)


def plot_sensitivity(
    sensitivity_results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    Plot sensitivity analysis results as a grid of line plots.

    Args:
        sensitivity_results: Output from optimizer.sensitivity_analysis()
        figsize: Figure size
        save_path: Optional path to save figure
        show: Whether to display

    Returns:
        (fig, axes) tuple
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. Install with: pip install lenia_field[viz]")

    param_names = list(sensitivity_results.keys())
    n_params = len(param_names)
    n_cols = min(4, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_params == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, name in enumerate(param_names):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        data = sensitivity_results[name]
        ax.plot(data["values"], data["fitness"], "b-o", linewidth=1.5, markersize=4)
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel("Fitness", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Mark maximum
        best_idx = np.argmax(data["fitness"])
        ax.axvline(data["values"][best_idx], color="r", linestyle="--", alpha=0.5)

    # Hide empty subplots
    for idx in range(n_params, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle("Parameter Sensitivity Analysis", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes


def plot_sensitivity_heatmap(
    optimizer: LeniaOptimizer,
    trajectories: np.ndarray,
    param1: str,
    param2: str,
    n_points: int = 10,
    base_params: Optional[Dict[str, float]] = None,
    n_warmup: int = 50,
    n_eval: int = 100,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    Create a 2D heatmap showing fitness as function of two parameters.

    Args:
        optimizer: LeniaOptimizer instance
        trajectories: Agent positions
        param1: First parameter name (x-axis)
        param2: Second parameter name (y-axis)
        n_points: Grid resolution
        base_params: Base parameter values
        n_warmup: Warmup steps
        n_eval: Evaluation steps
        figsize: Figure size
        save_path: Optional save path
        show: Whether to display

    Returns:
        (fig, ax, fitness_grid) tuple
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. Install with: pip install lenia_field[viz]")

    if base_params is None:
        base_params = {n: optimizer.parameter_specs[n].default for n in optimizer.param_names}

    spec1 = optimizer.parameter_specs[param1]
    spec2 = optimizer.parameter_specs[param2]

    values1 = np.linspace(spec1.bounds[0], spec1.bounds[1], n_points)
    values2 = np.linspace(spec2.bounds[0], spec2.bounds[1], n_points)

    fitness_grid = np.zeros((n_points, n_points))

    if optimizer.verbose:
        print(f"Computing {n_points}x{n_points} = {n_points**2} evaluations...")

    for i, v1 in enumerate(values1):
        for j, v2 in enumerate(values2):
            params = base_params.copy()
            params[param1] = float(v1)
            params[param2] = float(v2)
            metrics = optimizer.evaluate(params, trajectories, n_warmup, n_eval)
            fitness_grid[j, i] = metrics.fitness

        if optimizer.verbose:
            print(f"  Row {i + 1}/{n_points} complete")

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        fitness_grid,
        extent=[values1[0], values1[-1], values2[0], values2[-1]],
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
    )

    ax.set_xlabel(param1, fontsize=12)
    ax.set_ylabel(param2, fontsize=12)
    ax.set_title(f"Fitness: {param1} vs {param2}", fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fitness", fontsize=10)

    # Mark maximum
    best_idx = np.unravel_index(np.argmax(fitness_grid), fitness_grid.shape)
    ax.plot(values1[best_idx[1]], values2[best_idx[0]], "k*", markersize=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax, fitness_grid


def optimize_from_simulation(
    n_agents: int = 10,
    n_frames: int = 500,
    width: int = 512,
    height: int = 512,
    n_iterations: int = 100,
    seed: str = "default",
    save_preset: bool = True,
    preset_name: str = "optimized",
    verbose: bool = True,
    **optimizer_kwargs,
) -> OptimizationResult:
    """
    Run optimizer using simulation-generated trajectories.

    This is a convenience function that:
    1. Creates a Simulation with random agents
    2. Runs the simulation to collect trajectory data
    3. Optimizes Lenia parameters on those trajectories
    4. Optionally saves the result as a preset

    The optimized parameters can then be used anywhere:
    - LeniaField(get_optimized_config())
    - Simulation with optimized field_config
    - LeniaClient with optimized parameters
    - Any custom application

    Args:
        n_agents: Number of agents in simulation
        n_frames: Number of frames to collect (default 500 for robust optimization)
        width: Field width
        height: Field height
        n_iterations: Number of optimization iterations
        seed: Initialization seed for optimization:
            - "default": Use values from default.yaml as starting point
            - "random": Initialize randomly within parameter bounds
            - preset name (e.g., "optimized"): Use that preset's values
        save_preset: Whether to save result as a preset
        preset_name: Name for the saved preset
        verbose: Print progress information
        **optimizer_kwargs: Additional arguments for LeniaOptimizer.optimize()
            (e.g., population_size, mutation_rate, early_stop_patience)

    Returns:
        OptimizationResult with optimized parameters

    Example:
        # Run optimization from default.yaml values
        result = optimize_from_simulation(
            n_agents=10,
            n_frames=500,
            n_iterations=50,
        )

        # Or continue from a previous optimization
        result = optimize_from_simulation(
            seed="optimized",  # Use previous result as starting point
            n_iterations=50,
        )

        # View results
        print(result.summary())
        result.plot_convergence()

        # Use optimized parameters
        from lenia_field import get_optimized_config
        config = get_optimized_config()  # loads the saved preset
        field = LeniaField(config)
    """
    # Import simulation components
    from .simulation import Agent, AgentConfig, Simulation, SimulationConfig

    if verbose:
        print(f"Generating {n_frames} frames with {n_agents} agents...")

    # Create simulation with default config (don't use LeniaClient for trajectory collection)
    field_config = FieldConfig(width=width, height=height)
    agent_config = AgentConfig()
    sim_config = SimulationConfig(
        field_config=field_config,
        agent_config=agent_config,
        num_agents=n_agents,
        use_client=False,  # Direct mode for trajectory collection
    )

    sim = Simulation(config=sim_config)

    # Collect trajectories
    trajectories = []
    try:
        for i in range(n_frames):
            _, positions = sim.step()
            trajectories.append(positions.copy())
            if verbose and (i + 1) % 50 == 0:
                print(f"  Collected {i + 1}/{n_frames} frames")
    finally:
        sim.close()

    trajectories = np.stack(trajectories)

    if verbose:
        print(f"Trajectory shape: {trajectories.shape}")
        print(f"\nStarting optimization...")

    # Run optimization
    optimizer = LeniaOptimizer(
        width=width,
        height=height,
        seed=seed,
        verbose=verbose,
        **optimizer_kwargs,
    )

    result = optimizer.optimize(trajectories, n_iterations=n_iterations)

    # Save preset
    if save_preset:
        from .presets import save_optimized_preset

        path = save_optimized_preset(result, name=preset_name)
        if verbose:
            print(f"\nPreset saved to: {path}")

    return result


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("Testing LeniaOptimizer...")

    # Create synthetic trajectory (random walk)
    np.random.seed(42)
    n_frames = 200
    n_agents = 10
    trajectories = np.zeros((n_frames, n_agents, 2), dtype=np.float32)
    trajectories[0] = np.random.rand(n_agents, 2) * np.array([400, 400]) + 56

    for i in range(1, n_frames):
        step = np.random.randn(n_agents, 2) * 2
        trajectories[i] = np.clip(trajectories[i - 1] + step, 20, 492)

    # Quick optimization test
    print("\nRunning quick optimization (20 iterations)...")
    result = quick_optimize(trajectories, width=256, height=256, n_iterations=20)

    print(result.summary())

    # Save results
    result.save("test_optimization.json")
    print("\nResults saved to test_optimization.json")

    # Try plotting if matplotlib available
    try:
        print("\nGenerating plots...")
        result.plot_convergence(show=False, save_path="test_convergence.png")
        print("Saved test_convergence.png")
    except ImportError:
        print("matplotlib not available, skipping plots")
