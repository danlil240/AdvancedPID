"""
Unit tests for PID Controller.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import (
    PIDParams, 
    AntiWindupMethod, 
    DerivativeMode,
    ControllerType
)


class TestPIDController:
    """Test suite for PIDController class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        pid = PIDController()
        assert pid.params.kp == 1.0
        assert pid.params.ki == 0.0
        assert pid.params.kd == 0.0
    
    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        params = PIDParams(kp=2.0, ki=0.5, kd=0.1)
        pid = PIDController(params)
        assert pid.params.kp == 2.0
        assert pid.params.ki == 0.5
        assert pid.params.kd == 0.1
    
    def test_proportional_only(self):
        """Test P-only controller."""
        params = PIDParams(kp=2.0, ki=0.0, kd=0.0)
        pid = PIDController(params)
        
        # With setpoint=100, measurement=80, error=20
        # P term = 2.0 * 20 = 40
        output = pid.update(setpoint=100.0, measurement=80.0)
        assert abs(output - 40.0) < 1e-6
    
    def test_integral_accumulation(self):
        """Test integral term accumulates correctly."""
        params = PIDParams(kp=0.0, ki=1.0, kd=0.0, sample_time=0.1)
        pid = PIDController(params)
        
        # Each step with error=10 adds 1.0 to integral (ki*error*dt = 1.0*10*0.1)
        for i in range(10):
            output = pid.update(setpoint=10.0, measurement=0.0)
        
        # After 10 steps: integral = 10 * 1.0 = 10.0
        assert abs(pid.integral - 10.0) < 1e-6
    
    def test_output_saturation(self):
        """Test output saturation limits."""
        params = PIDParams(
            kp=10.0, ki=0.0, kd=0.0,
            output_min=-5.0, output_max=5.0
        )
        pid = PIDController(params)
        
        # Large error should saturate output
        output = pid.update(setpoint=100.0, measurement=0.0)
        assert output == 5.0
        
        output = pid.update(setpoint=0.0, measurement=100.0)
        assert output == -5.0
    
    def test_anti_windup_clamping(self):
        """Test anti-windup clamping method."""
        params = PIDParams(
            kp=1.0, ki=1.0, kd=0.0,
            sample_time=0.1,
            output_min=-10.0, output_max=10.0,
            anti_windup=AntiWindupMethod.CLAMPING
        )
        pid = PIDController(params)
        
        # Run with saturation
        for _ in range(100):
            pid.update(setpoint=100.0, measurement=0.0)
        
        # Integral should not grow unbounded
        assert pid.integral < 100.0  # Would be 1000 without anti-windup
    
    def test_anti_windup_back_calculation(self):
        """Test anti-windup back-calculation method."""
        params = PIDParams(
            kp=1.0, ki=1.0, kd=0.0,
            sample_time=0.1,
            output_min=-10.0, output_max=10.0,
            anti_windup=AntiWindupMethod.BACK_CALCULATION,
            back_calculation_gain=1.0
        )
        pid = PIDController(params)
        
        # Run with saturation
        for _ in range(100):
            pid.update(setpoint=100.0, measurement=0.0)
        
        # Integral should be limited
        assert pid.integral < 50.0
    
    def test_derivative_on_measurement(self):
        """Test derivative on measurement to avoid kick."""
        params = PIDParams(
            kp=0.0, ki=0.0, kd=1.0,
            derivative_mode=DerivativeMode.MEASUREMENT,
            derivative_filter_coeff=100.0  # Minimal filtering
        )
        pid = PIDController(params)
        
        # First update (initialization)
        pid.update(setpoint=0.0, measurement=50.0)
        
        # Step change in setpoint - should NOT cause derivative kick
        output = pid.update(setpoint=100.0, measurement=50.0)
        
        # D term should be based on measurement change (0), not setpoint change
        # With derivative on measurement, D = -kd * d(measurement)/dt
        assert abs(output) < 5.0  # Should be small, not a huge kick
    
    def test_setpoint_weighting(self):
        """Test setpoint weighting for reduced overshoot."""
        params = PIDParams(
            kp=2.0, ki=0.0, kd=0.0,
            setpoint_weight_p=0.5
        )
        pid = PIDController(params)
        
        # With b=0.5: error_p = 0.5*setpoint - measurement
        # = 0.5*100 - 0 = 50
        # P term = 2.0 * 50 = 100
        output = pid.update(setpoint=100.0, measurement=0.0)
        assert abs(output - 100.0) < 1e-6
    
    def test_reset(self):
        """Test controller reset."""
        params = PIDParams(kp=1.0, ki=1.0, kd=0.0, sample_time=0.1)
        pid = PIDController(params)
        
        # Build up state
        for _ in range(10):
            pid.update(setpoint=100.0, measurement=0.0)
        
        assert pid.integral > 0
        
        # Reset
        pid.reset()
        
        assert pid.integral == 0.0
        assert pid.output == 0.0
    
    def test_set_gains_bumpless(self):
        """Test bumpless parameter transfer."""
        params = PIDParams(kp=1.0, ki=0.5, kd=0.0)
        pid = PIDController(params)
        
        # Run to steady state
        for _ in range(100):
            pid.update(setpoint=50.0, measurement=50.0)
        
        old_output = pid.output
        
        # Change gains with bumpless transfer
        pid.set_gains(kp=2.0, ki=1.0, bumpless=True)
        
        # Output should remain similar
        new_output = pid.update(setpoint=50.0, measurement=50.0)
        assert abs(new_output - old_output) < 1.0
    
    def test_error_deadband(self):
        """Test error deadband."""
        params = PIDParams(
            kp=1.0, ki=0.0, kd=0.0,
            error_deadband=5.0
        )
        pid = PIDController(params)
        
        # Error of 3 is within deadband
        output = pid.update(setpoint=100.0, measurement=97.0)
        assert output == 0.0
        
        # Error of 10 is outside deadband
        output = pid.update(setpoint=100.0, measurement=90.0)
        assert output != 0.0
    
    def test_context_manager(self):
        """Test context manager protocol."""
        params = PIDParams(kp=1.0)
        
        with PIDController(params) as pid:
            output = pid.update(setpoint=100.0, measurement=50.0)
            assert output != 0
        
        # Should be closed after context


class TestPIDParams:
    """Test suite for PIDParams class."""
    
    def test_default_values(self):
        """Test default parameter values."""
        params = PIDParams()
        assert params.kp == 1.0
        assert params.ki == 0.0
        assert params.kd == 0.0
        assert params.sample_time == 0.01
    
    def test_validation_negative_kp(self):
        """Test validation rejects negative kp."""
        with pytest.raises(ValueError):
            PIDParams(kp=-1.0)
    
    def test_validation_invalid_output_limits(self):
        """Test validation of output limits."""
        with pytest.raises(ValueError):
            PIDParams(output_min=10.0, output_max=5.0)
    
    def test_validation_invalid_setpoint_weight(self):
        """Test validation of setpoint weight."""
        with pytest.raises(ValueError):
            PIDParams(setpoint_weight_p=1.5)
    
    def test_copy(self):
        """Test parameter copying."""
        params1 = PIDParams(kp=1.0, ki=0.5)
        params2 = params1.copy(kp=2.0)
        
        assert params1.kp == 1.0
        assert params2.kp == 2.0
        assert params2.ki == 0.5
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        params = PIDParams(kp=2.0, ki=0.5)
        d = params.to_dict()
        
        assert d['kp'] == 2.0
        assert d['ki'] == 0.5
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {'kp': 3.0, 'ki': 1.0, 'kd': 0.5}
        params = PIDParams.from_dict(d)
        
        assert params.kp == 3.0
        assert params.ki == 1.0
        assert params.kd == 0.5
    
    def test_json_serialization(self):
        """Test JSON serialization roundtrip."""
        params1 = PIDParams(kp=2.0, ki=0.5, kd=0.1)
        json_str = params1.to_json()
        params2 = PIDParams.from_json(json_str)
        
        assert params1.kp == params2.kp
        assert params1.ki == params2.ki
        assert params1.kd == params2.kd


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
