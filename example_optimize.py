#!/usr/bin/env python
"""
Example: Optimize Lenia field parameters from simulation trajectories.

This example demonstrates:
1. Running optimization using simulation-generated trajectories
2. Saving optimized parameters as a preset
3. Using the optimized parameters with different components

The optimized parameters are general-purpose and can be used with:
- LeniaField (direct mode)
- LeniaClient (subprocess mode)
- Simulation (with agents)
- Any custom application
"""

import numpy as np


def main():
    # Import from the package
    from lenia_field import (
        LeniaField,
        Simulation,
        SimulationConfig,
        get_optimized_config,
        list_presets,
        optimize_from_simulation,
    )

    print("=" * 60)
    print("LENIA PARAMETER OPTIMIZATION EXAMPLE")
    print("=" * 60)

    # =========================================================================
    # Step 1: Run optimization (generates trajectories internally)
    # =========================================================================
    print("\n[Step 1] Running optimization...")
    print("This generates trajectories from a simulation and finds optimal parameters.\n")

    result = optimize_from_simulation(
        n_agents=10,
        n_frames=300,  # Enough frames for robust trajectory data
        width=256,
        height=256,
        n_iterations=50,  # Enough iterations to converge
        save_preset=True,
        preset_name="example_optimized",
        verbose=True,
    )

    # Print results
    print("\n" + result.summary())

    # =========================================================================
    # Step 2: List available presets
    # =========================================================================
    print("\n[Step 2] Available presets:")
    presets = list_presets()
    for p in presets:
        print(f"  - {p}")

    # =========================================================================
    # Step 3: Use optimized parameters with LeniaField
    # =========================================================================
    print("\n[Step 3] Using optimized parameters with LeniaField...")

    # Load the optimized config
    config = get_optimized_config("example_optimized")
    print(f"  Loaded config: growth_mu={config.growth_mu:.4f}, decay={config.decay:.4f}")

    # Create field with optimized parameters
    field = LeniaField(config)

    # Run enough steps for field to build up (warmup)
    positions = np.array([[128, 128], [64, 64], [192, 192]], dtype=np.float32)
    for i in range(100):
        result_field = field.step(positions=positions)

    print(f"  After 100 steps: field_max={result_field.max():.4f}, field_mean={result_field.mean():.4f}")

    # =========================================================================
    # Step 4: Use optimized parameters with Simulation
    # =========================================================================
    print("\n[Step 4] Using optimized parameters with Simulation...")

    # Create simulation with optimized field config
    optimized_config = get_optimized_config("example_optimized")
    sim_config = SimulationConfig(
        field_config=optimized_config,
        num_agents=5,
        use_client=False,
    )
    sim = Simulation(config=sim_config)

    # Run simulation (enough steps for field to stabilize)
    for i in range(100):
        sim_field, positions = sim.step()

    stats = sim.get_stats()
    print(f"  After 100 steps: field_mean={stats['field_mean']:.4f}, num_agents={stats['num_agents']}")
    sim.close()

    # =========================================================================
    # Step 5: Using result directly (without saving as preset)
    # =========================================================================
    print("\n[Step 5] Using optimization result directly...")

    # You can also create a config directly from the result
    config_from_result = result.to_field_config(width=512, height=512)
    print(f"  Created config with custom size: {config_from_result.width}x{config_from_result.height}")
    print(f"  Parameters: growth_mu={config_from_result.growth_mu:.4f}, decay={config_from_result.decay:.4f}")

    # =========================================================================
    # Done!
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. optimize_from_simulation() generates trajectories and optimizes parameters")
    print("  2. Optimized parameters are saved as presets for later use")
    print("  3. get_optimized_config() loads a preset as a FieldConfig")
    print("  4. The config works with any component (LeniaField, Simulation, etc.)")
    print("\nTo visualize results (requires matplotlib):")
    print("  result.plot_convergence()")
    print("  result.plot_parameter_importance()")


if __name__ == "__main__":
    main()
