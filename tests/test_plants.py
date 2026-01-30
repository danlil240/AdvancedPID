"""
Unit tests for Plant models.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pid_control.plants.first_order import FirstOrderPlant
from pid_control.plants.second_order import SecondOrderPlant
from pid_control.plants.delay_plant import FOPDTPlant, DelayPlant
from pid_control.plants.nonlinear import NonlinearPlant, FrictionPlant


class TestFirstOrderPlant:
    """Test suite for FirstOrderPlant."""
    
    def test_initialization(self):
        """Test plant initialization."""
        plant = FirstOrderPlant(gain=2.0, time_constant=1.0)
        assert plant.gain == 2.0
        assert plant.time_constant == 1.0
    
    def test_step_response_final_value(self):
        """Test step response reaches correct final value."""
        plant = FirstOrderPlant(gain=2.0, time_constant=1.0, sample_time=0.01)
        
        # Apply step input of 1.0 for long time
        for _ in range(1000):  # 10 seconds
            output = plant.update(1.0)
        
        # Should reach gain * input = 2.0
        assert abs(output - 2.0) < 0.01
    
    def test_time_constant(self):
        """Test time constant behavior."""
        plant = FirstOrderPlant(gain=1.0, time_constant=1.0, sample_time=0.01)
        
        # After 1 time constant, should reach ~63.2% of final value
        for _ in range(100):  # 1 second
            output = plant.update(1.0)
        
        assert abs(output - 0.632) < 0.05
    
    def test_reset(self):
        """Test plant reset."""
        plant = FirstOrderPlant(gain=2.0, time_constant=1.0)
        
        for _ in range(100):
            plant.update(1.0)
        
        plant.reset()
        assert plant.output == 0.0
        assert plant.time == 0.0
    
    def test_noise(self):
        """Test noise addition."""
        plant = FirstOrderPlant(gain=1.0, time_constant=1.0)
        plant.set_noise(0.1)
        
        outputs = []
        for _ in range(100):
            outputs.append(plant.update(1.0))
        
        # With noise, outputs should vary
        assert np.std(outputs) > 0.01


class TestSecondOrderPlant:
    """Test suite for SecondOrderPlant."""
    
    def test_initialization(self):
        """Test plant initialization."""
        plant = SecondOrderPlant(
            gain=1.0,
            natural_frequency=2.0,
            damping_ratio=0.7
        )
        assert plant.gain == 1.0
        assert plant.natural_frequency == 2.0
        assert plant.damping_ratio == 0.7
    
    def test_underdamped_oscillation(self):
        """Test underdamped system oscillates."""
        plant = SecondOrderPlant(
            gain=1.0,
            natural_frequency=2.0,
            damping_ratio=0.2,  # Underdamped
            sample_time=0.01
        )
        
        outputs = []
        for _ in range(500):
            outputs.append(plant.update(1.0))
        
        # Should have overshoot
        assert max(outputs) > 1.0
    
    def test_critically_damped(self):
        """Test critically damped system."""
        plant = SecondOrderPlant(
            gain=1.0,
            natural_frequency=2.0,
            damping_ratio=1.0,  # Critically damped
            sample_time=0.01
        )
        
        outputs = []
        for _ in range(500):
            outputs.append(plant.update(1.0))
        
        # Should not overshoot significantly
        assert max(outputs) < 1.05
    
    def test_overdamped(self):
        """Test overdamped system."""
        plant = SecondOrderPlant(
            gain=1.0,
            natural_frequency=2.0,
            damping_ratio=2.0,  # Overdamped
            sample_time=0.01
        )
        
        outputs = []
        for _ in range(500):
            outputs.append(plant.update(1.0))
        
        # Should be monotonically increasing (no oscillation)
        for i in range(1, len(outputs)):
            assert outputs[i] >= outputs[i-1] - 1e-6
    
    def test_characteristic_times(self):
        """Test characteristic time calculations."""
        plant = SecondOrderPlant(
            gain=1.0,
            natural_frequency=2.0,
            damping_ratio=0.5
        )
        
        times = plant.get_characteristic_times()
        assert 'damped_frequency' in times
        assert 'overshoot_percent' in times
        assert times['overshoot_percent'] > 0


class TestFOPDTPlant:
    """Test suite for FOPDT plant."""
    
    def test_initialization(self):
        """Test plant initialization."""
        plant = FOPDTPlant(gain=2.0, time_constant=1.0, dead_time=0.5)
        assert plant.gain == 2.0
        assert plant.time_constant == 1.0
        assert plant.dead_time == 0.5
    
    def test_dead_time_delay(self):
        """Test dead time creates delay in response."""
        plant = FOPDTPlant(
            gain=1.0,
            time_constant=1.0,
            dead_time=0.5,
            sample_time=0.01
        )
        
        # Should not respond during dead time
        for i in range(40):  # 0.4 seconds
            output = plant.update(1.0)
            assert abs(output) < 0.01
        
        # After dead time, should start responding
        for _ in range(100):
            output = plant.update(1.0)
        
        assert output > 0.1
    
    def test_tuning_suggestions(self):
        """Test tuning suggestion generation."""
        plant = FOPDTPlant(gain=2.0, time_constant=3.0, dead_time=0.5)
        suggestions = plant.get_tuning_suggestions()
        
        assert 'ziegler_nichols' in suggestions
        assert 'cohen_coon' in suggestions
        assert 'imc_aggressive' in suggestions
        
        # Check reasonable values
        zn = suggestions['ziegler_nichols']
        assert zn['kp'] > 0
        assert zn['ki'] > 0


class TestNonlinearPlant:
    """Test suite for nonlinear plant."""
    
    def test_saturation(self):
        """Test input saturation."""
        plant = NonlinearPlant(
            gain=1.0,
            time_constant=0.5,
            saturation_limits=(-5, 5),
            sample_time=0.01
        )
        
        # Large input should be saturated
        for _ in range(200):
            plant.update(100.0)
        
        # Output limited by saturated input
        assert plant.output < 10.0
    
    def test_dead_zone(self):
        """Test dead zone nonlinearity."""
        plant = NonlinearPlant(
            gain=1.0,
            time_constant=0.1,
            dead_zone=2.0,
            sample_time=0.01
        )
        
        # Small input within dead zone
        for _ in range(100):
            output = plant.update(1.0)
        
        # Should produce no output
        assert abs(output) < 0.1


class TestFrictionPlant:
    """Test suite for friction plant."""
    
    def test_stiction(self):
        """Test static friction (stiction)."""
        plant = FrictionPlant(
            mass=1.0,
            stiction=5.0,
            coulomb_friction=3.0,
            viscous_friction=0.5,
            sample_time=0.01
        )
        
        # Small force below stiction
        for _ in range(100):
            output = plant.update(3.0)
        
        # Should not move
        assert abs(plant.position) < 0.1
        
        # Force above stiction
        plant.reset()
        for _ in range(100):
            output = plant.update(10.0)
        
        # Should move
        assert abs(plant.position) > 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
