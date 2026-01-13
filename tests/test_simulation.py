"""Tests for simulation.py module."""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lenia_field.simulation import Agent, AgentConfig, Simulation, SimulationConfig
from lenia_field.core import FieldConfig


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_values(self):
        config = AgentConfig()
        assert config.speed == 2.0
        assert config.turn_rate == 0.3
        assert config.persistence == 0.8
        assert config.boundary_mode == "reflect"
        assert config.injection_power == 2.0  # Must be > 1.0 to overcome negative Lenia growth
        assert config.injection_enabled is True

    def test_custom_values(self):
        config = AgentConfig(speed=5.0, persistence=0.5, boundary_mode="wrap")
        assert config.speed == 5.0
        assert config.persistence == 0.5
        assert config.boundary_mode == "wrap"

    def test_color_and_radius(self):
        config = AgentConfig(color=(255, 0, 0), radius=8)
        assert config.color == (255, 0, 0)
        assert config.radius == 8


class TestAgent:
    """Tests for Agent class."""

    def test_initialization_default(self):
        agent = Agent(x=100, y=100)
        assert np.allclose(agent.position, [100, 100])
        assert 0 <= agent.heading < 2 * np.pi
        assert agent.id == 0

    def test_initialization_custom_config(self):
        config = AgentConfig(speed=5.0)
        agent = Agent(x=50, y=50, config=config, agent_id=5)
        assert agent.config.speed == 5.0
        assert agent.id == 5

    def test_initialization_custom_heading(self):
        agent = Agent(x=100, y=100, initial_heading=np.pi / 2)
        assert np.isclose(agent.heading, np.pi / 2)

    def test_update_moves_agent(self):
        config = AgentConfig(speed=2.0, persistence=1.0)  # High persistence = straight line
        agent = Agent(x=100, y=100, config=config, initial_heading=0)
        initial_pos = agent.position.copy()

        agent.update(field_bounds=(200, 200))

        # Agent should have moved
        assert not np.allclose(agent.position, initial_pos)

        # With heading=0 and high persistence, should move roughly in +x direction
        assert agent.position[0] > initial_pos[0]

    def test_update_returns_position(self):
        agent = Agent(x=100, y=100)
        new_pos = agent.update(field_bounds=(200, 200))
        assert isinstance(new_pos, np.ndarray)
        assert new_pos.shape == (2,)
        assert np.allclose(new_pos, agent.position)

    def test_boundary_reflect_left(self):
        config = AgentConfig(boundary_mode="reflect", speed=10.0, boundary_margin=5.0)
        agent = Agent(x=3, y=100, config=config, initial_heading=np.pi)  # Heading left

        agent.update(field_bounds=(200, 200))

        # Should be pushed back into the field
        assert agent.position[0] >= config.boundary_margin

    def test_boundary_reflect_right(self):
        config = AgentConfig(boundary_mode="reflect", speed=10.0, boundary_margin=5.0)
        agent = Agent(x=197, y=100, config=config, initial_heading=0)  # Heading right

        agent.update(field_bounds=(200, 200))

        # Should be pushed back into the field
        assert agent.position[0] <= 200 - config.boundary_margin

    def test_boundary_reflect_top(self):
        config = AgentConfig(boundary_mode="reflect", speed=10.0, boundary_margin=5.0)
        agent = Agent(x=100, y=3, config=config, initial_heading=-np.pi/2)  # Heading up (negative y)

        agent.update(field_bounds=(200, 200))

        # Should be pushed back into the field
        assert agent.position[1] >= config.boundary_margin

    def test_boundary_reflect_bottom(self):
        config = AgentConfig(boundary_mode="reflect", speed=10.0, boundary_margin=5.0)
        agent = Agent(x=100, y=197, config=config, initial_heading=np.pi/2)  # Heading down

        agent.update(field_bounds=(200, 200))

        # Should be pushed back into the field
        assert agent.position[1] <= 200 - config.boundary_margin

    def test_boundary_wrap(self):
        config = AgentConfig(boundary_mode="wrap", speed=15.0)
        agent = Agent(x=195, y=100, config=config, initial_heading=0)  # Heading right

        agent.update(field_bounds=(200, 200))

        # Should wrap to left side
        assert 0 <= agent.position[0] < 200
        assert 0 <= agent.position[1] < 200

    def test_persistence_affects_path_variance(self):
        """Higher persistence should produce straighter paths."""
        np.random.seed(42)

        # Low persistence agent
        config_low = AgentConfig(persistence=0.1, speed=2.0)
        agent_low = Agent(x=100, y=100, config=config_low, initial_heading=0)

        # High persistence agent
        config_high = AgentConfig(persistence=0.95, speed=2.0)
        agent_high = Agent(x=100, y=100, config=config_high, initial_heading=0)

        # Run many steps and track heading changes
        headings_low = [agent_low.heading]
        headings_high = [agent_high.heading]

        for _ in range(50):
            agent_low.update((500, 500))
            agent_high.update((500, 500))
            headings_low.append(agent_low.heading)
            headings_high.append(agent_high.heading)

        # Calculate heading change variance
        changes_low = np.diff(headings_low)
        changes_high = np.diff(headings_high)

        var_low = np.var(changes_low)
        var_high = np.var(changes_high)

        # Low persistence should have more variance in heading changes
        assert var_low > var_high

    def test_velocity_property(self):
        agent = Agent(x=100, y=100, initial_heading=0)
        vel = agent.velocity

        assert isinstance(vel, np.ndarray)
        assert vel.shape == (2,)
        # With heading=0, velocity should be primarily in +x direction
        assert vel[0] > 0
        assert abs(vel[1]) < abs(vel[0])

    def test_set_position(self):
        agent = Agent(x=100, y=100)
        agent.set_position(50, 75)
        assert np.allclose(agent.position, [50, 75])

    def test_set_heading(self):
        agent = Agent(x=100, y=100)
        agent.set_heading(np.pi)
        assert np.isclose(agent.heading, np.pi)

        # Test normalization
        agent.set_heading(3 * np.pi)
        assert 0 <= agent.heading < 2 * np.pi

    def test_get_stats(self):
        agent = Agent(x=100, y=100)

        # Before any movement
        stats = agent.get_stats()
        assert stats["step_count"] == 0
        assert stats["total_distance"] == 0.0

        # After some movement
        for _ in range(10):
            agent.update((200, 200))

        stats = agent.get_stats()
        assert stats["step_count"] == 10
        assert stats["total_distance"] > 0
        assert stats["average_speed"] > 0


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_default_values(self):
        config = SimulationConfig()
        assert config.num_agents == 10
        assert config.use_client is False
        assert config.target_fps == 60
        assert isinstance(config.field_config, FieldConfig)
        assert isinstance(config.agent_config, AgentConfig)

    def test_custom_field_config(self):
        field_config = FieldConfig(width=256, height=256)
        config = SimulationConfig(field_config=field_config, num_agents=5)
        assert config.field_config.width == 256
        assert config.num_agents == 5


class TestSimulation:
    """Tests for Simulation class."""

    @pytest.fixture
    def small_sim_config(self):
        return SimulationConfig(
            field_config=FieldConfig(width=64, height=64),
            num_agents=5,
            use_client=False
        )

    def test_initialization(self, small_sim_config):
        sim = Simulation(config=small_sim_config)
        try:
            assert len(sim.agents) == 5
            assert sim.field is None  # Before first step
            assert sim.fps == 0.0
            assert sim.is_paused is False
        finally:
            sim.close()

    def test_initialization_with_custom_agents(self, small_sim_config):
        agents = [
            Agent(x=10, y=10, agent_id=0),
            Agent(x=20, y=20, agent_id=1),
        ]
        sim = Simulation(config=small_sim_config, agents=agents)
        try:
            assert len(sim.agents) == 2
            assert sim.agents[0].id == 0
            assert sim.agents[1].id == 1
        finally:
            sim.close()

    def test_step_returns_field_and_positions(self, small_sim_config):
        sim = Simulation(config=small_sim_config)
        try:
            field, positions = sim.step()

            assert isinstance(field, np.ndarray)
            assert field.shape == (64, 64)
            assert field.dtype == np.float32 or field.dtype == np.float64

            assert isinstance(positions, np.ndarray)
            assert positions.shape == (5, 2)
        finally:
            sim.close()

    def test_step_updates_field(self, small_sim_config):
        # Use higher injection power to see effect
        small_sim_config.agent_config.injection_power = 0.5
        sim = Simulation(config=small_sim_config)
        try:
            field1, _ = sim.step()
            field2, _ = sim.step()

            # Field should change between steps
            assert sim.field is not None
        finally:
            sim.close()

    def test_add_agent(self, small_sim_config):
        sim = Simulation(config=small_sim_config)
        try:
            initial_count = len(sim.agents)
            new_agent = sim.add_agent(32, 32)

            assert len(sim.agents) == initial_count + 1
            assert new_agent in sim.agents
            assert np.allclose(new_agent.position, [32, 32])
        finally:
            sim.close()

    def test_remove_agent(self, small_sim_config):
        sim = Simulation(config=small_sim_config)
        try:
            agent_id = sim.agents[0].id
            initial_count = len(sim.agents)

            result = sim.remove_agent(agent_id)

            assert result is True
            assert len(sim.agents) == initial_count - 1
            assert all(a.id != agent_id for a in sim.agents)
        finally:
            sim.close()

    def test_remove_nonexistent_agent(self, small_sim_config):
        sim = Simulation(config=small_sim_config)
        try:
            result = sim.remove_agent(9999)
            assert result is False
        finally:
            sim.close()

    def test_reset(self, small_sim_config):
        sim = Simulation(config=small_sim_config)
        try:
            # Run a few steps
            sim.step()
            sim.step()
            initial_positions = sim.get_positions().copy()

            # Reset
            sim.reset()

            # Positions should be different (randomized)
            new_positions = sim.get_positions()
            assert not np.allclose(initial_positions, new_positions)

            # Field should be reset
            assert sim.field is None
        finally:
            sim.close()

    def test_pause_resume(self, small_sim_config):
        sim = Simulation(config=small_sim_config)
        try:
            assert sim.is_paused is False

            sim.pause()
            assert sim.is_paused is True

            sim.resume()
            assert sim.is_paused is False
        finally:
            sim.close()

    def test_toggle_pause(self, small_sim_config):
        sim = Simulation(config=small_sim_config)
        try:
            assert sim.is_paused is False

            sim.toggle_pause()
            assert sim.is_paused is True

            sim.toggle_pause()
            assert sim.is_paused is False
        finally:
            sim.close()

    def test_get_positions(self, small_sim_config):
        sim = Simulation(config=small_sim_config)
        try:
            positions = sim.get_positions()

            assert isinstance(positions, np.ndarray)
            assert positions.shape == (5, 2)
            assert positions.dtype == np.float32
        finally:
            sim.close()

    def test_get_stats(self, small_sim_config):
        sim = Simulation(config=small_sim_config)
        try:
            # Before stepping
            stats = sim.get_stats()
            assert stats["step_count"] == 0
            assert stats["num_agents"] == 5

            # After stepping
            sim.step()
            sim.step()

            stats = sim.get_stats()
            assert stats["step_count"] == 2
            assert "field_max" in stats
            assert "field_mean" in stats
        finally:
            sim.close()

    def test_empty_simulation(self):
        config = SimulationConfig(
            field_config=FieldConfig(width=32, height=32),
            num_agents=0,
            use_client=False
        )
        sim = Simulation(config=config)
        try:
            assert len(sim.agents) == 0

            field, positions = sim.step()

            assert field.shape == (32, 32)
            assert positions.shape == (0, 2)
        finally:
            sim.close()


class TestAgentBoundaryBehavior:
    """Additional tests for agent boundary behavior."""

    def test_agent_stays_in_bounds_reflect(self):
        """Agent should always stay within field bounds in reflect mode."""
        config = AgentConfig(boundary_mode="reflect", speed=5.0)
        agent = Agent(x=100, y=100, config=config)
        bounds = (200, 200)

        for _ in range(1000):
            agent.update(bounds)
            pos = agent.position
            assert 0 <= pos[0] < bounds[0], f"x={pos[0]} out of bounds"
            assert 0 <= pos[1] < bounds[1], f"y={pos[1]} out of bounds"

    def test_agent_stays_in_bounds_wrap(self):
        """Agent should always stay within field bounds in wrap mode."""
        config = AgentConfig(boundary_mode="wrap", speed=5.0)
        agent = Agent(x=100, y=100, config=config)
        bounds = (200, 200)

        for _ in range(1000):
            agent.update(bounds)
            pos = agent.position
            assert 0 <= pos[0] < bounds[0], f"x={pos[0]} out of bounds"
            assert 0 <= pos[1] < bounds[1], f"y={pos[1]} out of bounds"


class TestSimulationPerformance:
    """Performance-related tests."""

    def test_many_agents(self):
        """Test simulation with many agents."""
        config = SimulationConfig(
            field_config=FieldConfig(width=128, height=128),
            num_agents=100,
            use_client=False
        )
        sim = Simulation(config=config)
        try:
            assert len(sim.agents) == 100

            # Should complete without error
            for _ in range(10):
                field, positions = sim.step()
                assert positions.shape == (100, 2)
        finally:
            sim.close()

    def test_rapid_add_remove(self):
        """Test rapidly adding and removing agents."""
        config = SimulationConfig(
            field_config=FieldConfig(width=64, height=64),
            num_agents=5,
            use_client=False
        )
        sim = Simulation(config=config)
        try:
            for i in range(20):
                agent = sim.add_agent(32, 32)
                sim.step()
                sim.remove_agent(agent.id)

            # Should still have original agents
            assert len(sim.agents) == 5
        finally:
            sim.close()
