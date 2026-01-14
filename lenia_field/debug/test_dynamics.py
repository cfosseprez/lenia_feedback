"""
Debug script to visualize and test Lenia dynamics components.

Run: python -m lenia_field.debug.test_dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lenia_field.core import LeniaField, FieldConfig, growth_function, make_kernel


def test_growth_function():
    """Visualize the growth function across potential values."""
    print("\n=== Testing Growth Function ===")

    # Test with different mu/sigma values
    potentials = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Current parameters
    mu, sigma = 0.10, 0.03
    growth = 2.0 * np.exp(-((potentials - mu) ** 2) / (2 * sigma ** 2)) - 1.0
    axes[0].plot(potentials, growth, 'b-', linewidth=2)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].axvline(x=mu, color='g', linestyle='--', alpha=0.5, label=f'mu={mu}')
    axes[0].set_title(f'Current: mu={mu}, sigma={sigma}')
    axes[0].set_xlabel('Potential (convolution result)')
    axes[0].set_ylabel('Growth')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Paper parameters
    mu, sigma = 0.15, 0.015
    growth = 2.0 * np.exp(-((potentials - mu) ** 2) / (2 * sigma ** 2)) - 1.0
    axes[1].plot(potentials, growth, 'b-', linewidth=2)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].axvline(x=mu, color='g', linestyle='--', alpha=0.5, label=f'mu={mu}')
    axes[1].set_title(f'Paper: mu={mu}, sigma={sigma}')
    axes[1].set_xlabel('Potential (convolution result)')
    axes[1].set_ylabel('Growth')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Wider sigma
    mu, sigma = 0.15, 0.05
    growth = 2.0 * np.exp(-((potentials - mu) ** 2) / (2 * sigma ** 2)) - 1.0
    axes[2].plot(potentials, growth, 'b-', linewidth=2)
    axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[2].axvline(x=mu, color='g', linestyle='--', alpha=0.5, label=f'mu={mu}')
    axes[2].set_title(f'Wider: mu={mu}, sigma={sigma}')
    axes[2].set_xlabel('Potential (convolution result)')
    axes[2].set_ylabel('Growth')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('debug_growth_function.png', dpi=100)
    print("Saved: debug_growth_function.png")

    # Print key points
    print(f"\nGrowth function characteristics:")
    print(f"  - Returns +1 at potential = mu")
    print(f"  - Returns -1 when far from mu")
    print(f"  - Positive only in narrow band: mu ± ~3*sigma")
    print(f"  - With mu=0.15, sigma=0.015: positive range is [0.105, 0.195]")


def test_injection_magnitude():
    """Test how much injection contributes to field per step."""
    print("\n=== Testing Injection Magnitude ===")

    # Create field
    config = FieldConfig(width=64, height=64, suzuki_style=False)
    field = LeniaField(config)

    # Single agent at center
    positions = np.array([[32, 32]], dtype=np.float32)

    # Test different injection powers
    powers_to_test = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

    print(f"\nInjection test (dt={config.dt}, radius={config.injection_radius}):")
    print(f"{'Power':<10} {'Max Added':<12} {'Steps to 0.5':<15} {'Steps to 1.0':<15}")
    print("-" * 55)

    for power in powers_to_test:
        field.reset()
        powers = np.array([power], dtype=np.float32)

        # Run one step
        field.step(positions, powers)
        max_val = field.get_field().max()

        # Estimate steps to reach thresholds
        steps_to_half = 0.5 / max_val if max_val > 0 else float('inf')
        steps_to_one = 1.0 / max_val if max_val > 0 else float('inf')

        print(f"{power:<10.2f} {max_val:<12.4f} {steps_to_half:<15.1f} {steps_to_one:<15.1f}")


def test_resource_dynamics():
    """Test resource consumption and recovery."""
    print("\n=== Testing Resource Dynamics ===")

    # Create field with Suzuki mode
    config = FieldConfig(
        width=64, height=64,
        suzuki_style=True,
        R_C=0.005,  # Paper value
        R_G=0.002,  # Paper value
        r_max=1.0,
        r_initial=1.0,
        resource_effect_rate=0.5
    )
    field = LeniaField(config)

    # Single agent at center
    positions = np.array([[32, 32]], dtype=np.float32)
    powers = np.array([0.05], dtype=np.float32)

    # Track resource at agent location over time
    steps = 100
    resource_at_agent = []
    resource_mean = []
    field_at_agent = []
    field_mean = []

    for _ in range(steps):
        field.step(positions, powers)
        resource_at_agent.append(field.get_resource()[32, 32])
        resource_mean.append(field.get_resource().mean())
        field_at_agent.append(field.get_field()[32, 32])
        field_mean.append(field.get_field().mean())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(resource_at_agent, 'b-', label='At agent')
    axes[0, 0].plot(resource_mean, 'r--', label='Mean')
    axes[0, 0].set_title('Resource Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Resource')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(field_at_agent, 'b-', label='At agent')
    axes[0, 1].plot(field_mean, 'r--', label='Mean')
    axes[0, 1].set_title('Field Over Time')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Field value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Show final resource field
    im = axes[1, 0].imshow(field.get_resource(), cmap='viridis')
    axes[1, 0].set_title('Final Resource Field')
    plt.colorbar(im, ax=axes[1, 0])

    # Show final morphogen field
    im = axes[1, 1].imshow(field.get_field(), cmap='plasma')
    axes[1, 1].set_title('Final Morphogen Field')
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig('debug_resource_dynamics.png', dpi=100)
    print("Saved: debug_resource_dynamics.png")

    print(f"\nResource dynamics (R_C={config.R_C}, R_G={config.R_G}):")
    print(f"  - Resource at agent: {resource_at_agent[0]:.3f} -> {resource_at_agent[-1]:.3f}")
    print(f"  - Resource mean: {resource_mean[0]:.3f} -> {resource_mean[-1]:.3f}")
    print(f"  - Field at agent: {field_at_agent[0]:.4f} -> {field_at_agent[-1]:.4f}")
    print(f"  - Field mean: {field_mean[0]:.6f} -> {field_mean[-1]:.6f}")


def test_injection_vs_growth():
    """Compare injection contribution vs growth contribution."""
    print("\n=== Testing Injection vs Growth Balance ===")

    config = FieldConfig(
        width=64, height=64,
        suzuki_style=True,
        growth_mu=0.15,
        growth_sigma=0.015,
        decay=0.01,
        R_C=0.005,
        R_G=0.002
    )
    field = LeniaField(config)

    # Single agent
    positions = np.array([[32, 32]], dtype=np.float32)
    powers = np.array([0.05], dtype=np.float32)

    # Track contributions over time
    steps = 50
    injection_contrib = []
    growth_contrib = []
    decay_contrib = []
    net_change = []

    for i in range(steps):
        metrics = field.step_with_metrics(positions, powers)

        injection_contrib.append(metrics['injection_total'] * config.dt)
        growth_contrib.append((metrics['growth_positive'] + metrics['growth_negative']) * config.dt)
        decay_contrib.append(metrics['decay_total'] * config.dt)
        net_change.append(metrics['delta_field_mean'] * 64 * 64)  # Total change

    fig, ax = plt.subplots(figsize=(10, 6))

    steps_arr = np.arange(steps)
    ax.plot(steps_arr, injection_contrib, 'g-', label='Injection', linewidth=2)
    ax.plot(steps_arr, growth_contrib, 'b-', label='Growth (pos+neg)', linewidth=2)
    ax.plot(steps_arr, [-d for d in decay_contrib], 'r-', label='-Decay', linewidth=2)
    ax.plot(steps_arr, net_change, 'k--', label='Net change', linewidth=2)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Injection vs Growth vs Decay (per step)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Total contribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('debug_injection_vs_growth.png', dpi=100)
    print("Saved: debug_injection_vs_growth.png")

    print(f"\nBalance analysis:")
    print(f"  - Avg injection: {np.mean(injection_contrib):.4f}")
    print(f"  - Avg growth: {np.mean(growth_contrib):.4f}")
    print(f"  - Avg decay: {np.mean(decay_contrib):.4f}")
    print(f"  - Avg net: {np.mean(net_change):.6f}")


def test_kernel():
    """Visualize the Lenia kernel."""
    print("\n=== Testing Kernel ===")

    config = FieldConfig(kernel_radius=13, kernel_sigma=0.5)
    kernel = make_kernel(config)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im = axes[0].imshow(kernel, cmap='viridis')
    axes[0].set_title(f'Lenia Kernel (r={config.kernel_radius}, sigma={config.kernel_sigma})')
    plt.colorbar(im, ax=axes[0])

    # Cross-section
    center = config.kernel_radius
    axes[1].plot(kernel[center, :], 'b-', linewidth=2)
    axes[1].set_title('Kernel Cross-section')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Weight')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('debug_kernel.png', dpi=100)
    print("Saved: debug_kernel.png")

    print(f"\nKernel stats:")
    print(f"  - Shape: {kernel.shape}")
    print(f"  - Sum: {kernel.sum():.4f} (should be ~1.0)")
    print(f"  - Max: {kernel.max():.4f}")
    print(f"  - Ring shape: peaks at distance ~{config.kernel_radius * 0.5:.1f}")


def run_all_tests():
    """Run all debug tests."""
    print("=" * 60)
    print("LENIA DYNAMICS DEBUG SUITE")
    print("=" * 60)

    test_kernel()
    test_growth_function()
    test_injection_magnitude()
    test_resource_dynamics()
    test_injection_vs_growth()

    print("\n" + "=" * 60)
    print("All debug plots saved!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
