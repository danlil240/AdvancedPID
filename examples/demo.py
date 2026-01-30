from pid_control.plants.second_order import SecondOrderPlant
from pid_control.core.pid_controller import PIDController
from pid_control.core.pid_params import PIDParams

G = SecondOrderPlant(
    sample_time=0.1,
    gain=1.0,
    damping_ratio=0.707,
    initial_output=0.0,
    initial_velocity=0.0,
    natural_frequency=1.0
)


def main():
    print("=" * 60)
    print("Basic PID Controller Demo")
    print("=" * 60)

    pid_params = PIDParams(kp=1.0, ki=0.1, kd=0.01)
    pid_controller = PIDController(pid_params, G)
    pid_controller.
    