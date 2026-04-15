"""
Gymnasium environment wrappers for all PID control plants.

Each environment follows the standard Gymnasium API:
  obs, info = env.reset()
  obs, reward, terminated, truncated, info = env.step(action)
  env.render()   # when render_mode="human"

Observation: plant output (and optionally full state)
Action: scalar control input (force, voltage, etc.)
Reward: negative absolute error from setpoint

Render modes
------------
"human"  : live matplotlib window updated every step
None     : no rendering (default)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
from collections import deque

from pid_control.plants.base_plant import BasePlant
from pid_control.plants.first_order import FirstOrderPlant
from pid_control.plants.second_order import SecondOrderPlant
from pid_control.plants.nonlinear import NonlinearPlant, FrictionPlant
from pid_control.plants.delay_plant import FOPDTPlant, DelayPlant
from pid_control.plants.double_pendulum import DoublePendulumCart


def _tk_available() -> bool:
    try:
        import tkinter  # noqa: F401
        return True
    except ImportError:
        return False


# ======================================================================
# Base environment
# ======================================================================

class PIDPlantEnv(gym.Env):
    """
    Generic Gymnasium wrapper around any BasePlant.

    Render (render_mode="human")
    ----------------------------
    Opens a live 3-panel matplotlib window showing:
      - Top:    measurement vs setpoint over a rolling window
      - Middle: tracking error
      - Bottom: control output (action)

    Physics options
    ---------------
    ``set_disturbance(value)`` and ``set_noise(std)`` can be called at any
    time during an episode to inject physics perturbations into the plant.
    """

    metadata = {"render_modes": ["human"]}
    _RENDER_WINDOW = 300  # rolling history length (steps)

    def __init__(
        self,
        plant: BasePlant,
        setpoint: float = 0.0,
        max_steps: int = 1000,
        action_low: float = -100.0,
        action_high: float = 100.0,
        obs_low: float = -1e4,
        obs_high: float = 1e4,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.plant = plant
        self.setpoint = setpoint
        self.max_steps = max_steps
        self._step_count = 0
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=np.float32(action_low),
            high=np.float32(action_high),
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.float32(obs_low),
            high=np.float32(obs_high),
            shape=(1,),
            dtype=np.float32,
        )

        # Render state
        self._fig = None
        self._hist_t: deque = deque(maxlen=self._RENDER_WINDOW)
        self._hist_obs: deque = deque(maxlen=self._RENDER_WINDOW)
        self._hist_sp: deque = deque(maxlen=self._RENDER_WINDOW)
        self._hist_err: deque = deque(maxlen=self._RENDER_WINDOW)
        self._hist_ctrl: deque = deque(maxlen=self._RENDER_WINDOW)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.plant.reset()
        self._step_count = 0
        self._hist_t.clear()
        self._hist_obs.clear()
        self._hist_sp.clear()
        self._hist_err.clear()
        self._hist_ctrl.clear()
        obs = np.array([self.plant.output], dtype=np.float32)
        if self.render_mode == "human":
            self._init_render()
        return obs, self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        control = float(np.clip(action, self.action_space.low, self.action_space.high)[0])
        measurement = self.plant.update(control)
        self._step_count += 1

        self._hist_t.append(self.plant.time)
        self._hist_obs.append(measurement)
        self._hist_sp.append(self.setpoint)
        self._hist_err.append(self.setpoint - measurement)
        self._hist_ctrl.append(control)

        obs = np.array([measurement], dtype=np.float32)
        reward = -abs(self.setpoint - measurement)
        terminated = False
        truncated = self._step_count >= self.max_steps

        if self.render_mode == "human":
            self._update_render()

        return obs, reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            if self._fig is None:
                self._init_render()
            self._update_render()

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------
    def set_setpoint(self, sp: float) -> None:
        self.setpoint = sp

    def set_disturbance(self, value: float) -> None:
        self.plant.set_disturbance(value)

    def set_noise(self, std: float) -> None:
        self.plant.set_noise(std)

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------
    def _init_render(self):
        import matplotlib
        if _tk_available():
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        plt.ion()
        self._fig, (self._ax_out, self._ax_err, self._ax_ctrl) = plt.subplots(
            3, 1, figsize=(9, 7), sharex=True
        )
        self._fig.suptitle(f"{type(self).__name__}  —  live render", fontsize=12)

        self._ax_out.set_ylabel("Output")
        self._ax_out.grid(True, alpha=0.3)
        self._ax_err.set_ylabel("Error")
        self._ax_err.axhline(0, color="gray", lw=0.8, ls="--")
        self._ax_err.grid(True, alpha=0.3)
        self._ax_ctrl.set_ylabel("Control")
        self._ax_ctrl.set_xlabel("Time (s)")
        self._ax_ctrl.grid(True, alpha=0.3)

        (self._line_obs,) = self._ax_out.plot([], [], "b-", lw=1.5, label="Output")
        (self._line_sp,) = self._ax_out.plot([], [], "r--", lw=1.5, label="Setpoint")
        self._ax_out.legend(loc="upper left", fontsize=8)
        (self._line_err,) = self._ax_err.plot([], [], "m-", lw=1.2)
        (self._line_ctrl,) = self._ax_ctrl.plot([], [], "g-", lw=1.2)

        plt.tight_layout()
        self._fig.canvas.draw()

    def _update_render(self):
        import matplotlib.pyplot as plt
        if self._fig is None:
            return
        t = list(self._hist_t)
        if not t:
            return

        self._line_obs.set_data(t, list(self._hist_obs))
        self._line_sp.set_data(t, list(self._hist_sp))
        self._line_err.set_data(t, list(self._hist_err))
        self._line_ctrl.set_data(t, list(self._hist_ctrl))

        for ax in (self._ax_out, self._ax_err, self._ax_ctrl):
            ax.relim()
            ax.autoscale_view()

        self._fig.canvas.flush_events()
        plt.pause(0.001)

    # ------------------------------------------------------------------
    def _get_info(self) -> Dict[str, Any]:
        return {
            "time": self.plant.time,
            "output": self.plant.output,
            "step": self._step_count,
        }


# ======================================================================
# Concrete environments
# ======================================================================

class FirstOrderEnv(PIDPlantEnv):
    """Gymnasium env wrapping :class:`FirstOrderPlant`."""

    def __init__(
        self,
        gain: float = 1.0,
        time_constant: float = 1.0,
        sample_time: float = 0.01,
        initial_output: float = 0.0,
        setpoint: float = 0.0,
        max_steps: int = 1000,
        action_low: float = -100.0,
        action_high: float = 100.0,
        render_mode: Optional[str] = None,
    ):
        plant = FirstOrderPlant(
            gain=gain,
            time_constant=time_constant,
            sample_time=sample_time,
            initial_output=initial_output,
        )
        super().__init__(
            plant, setpoint=setpoint, max_steps=max_steps,
            action_low=action_low, action_high=action_high,
            render_mode=render_mode,
        )


class SecondOrderEnv(PIDPlantEnv):
    """Gymnasium env wrapping :class:`SecondOrderPlant`."""

    def __init__(
        self,
        gain: float = 1.0,
        natural_frequency: float = 1.0,
        damping_ratio: float = 0.7,
        sample_time: float = 0.01,
        initial_output: float = 0.0,
        initial_velocity: float = 0.0,
        setpoint: float = 0.0,
        max_steps: int = 1000,
        action_low: float = -100.0,
        action_high: float = 100.0,
        render_mode: Optional[str] = None,
    ):
        plant = SecondOrderPlant(
            gain=gain,
            natural_frequency=natural_frequency,
            damping_ratio=damping_ratio,
            sample_time=sample_time,
            initial_output=initial_output,
            initial_velocity=initial_velocity,
        )
        super().__init__(
            plant, setpoint=setpoint, max_steps=max_steps,
            action_low=action_low, action_high=action_high,
            render_mode=render_mode,
        )


class NonlinearEnv(PIDPlantEnv):
    """Gymnasium env wrapping :class:`NonlinearPlant`."""

    def __init__(
        self,
        gain: float = 1.0,
        time_constant: float = 1.0,
        sample_time: float = 0.01,
        saturation_limits=None,
        dead_zone: float = 0.0,
        backlash: float = 0.0,
        nonlinear_gain_func=None,
        initial_output: float = 0.0,
        setpoint: float = 0.0,
        max_steps: int = 1000,
        action_low: float = -100.0,
        action_high: float = 100.0,
        render_mode: Optional[str] = None,
    ):
        plant = NonlinearPlant(
            gain=gain,
            time_constant=time_constant,
            sample_time=sample_time,
            saturation_limits=saturation_limits,
            dead_zone=dead_zone,
            backlash=backlash,
            nonlinear_gain_func=nonlinear_gain_func,
            initial_output=initial_output,
        )
        super().__init__(
            plant, setpoint=setpoint, max_steps=max_steps,
            action_low=action_low, action_high=action_high,
            render_mode=render_mode,
        )


class FrictionPlantEnv(PIDPlantEnv):
    """
    Gymnasium env wrapping :class:`FrictionPlant`.

    Observation (2,): [position, velocity]
    Render: live plot with position/setpoint, velocity, and force with
            stiction threshold bands marked.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        mass: float = 1.0,
        viscous_friction: float = 0.5,
        coulomb_friction: float = 0.1,
        stiction: float = 0.15,
        sample_time: float = 0.01,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0,
        setpoint: float = 0.0,
        max_steps: int = 1000,
        action_low: float = -100.0,
        action_high: float = 100.0,
        render_mode: Optional[str] = None,
    ):
        plant = FrictionPlant(
            mass=mass,
            viscous_friction=viscous_friction,
            coulomb_friction=coulomb_friction,
            stiction=stiction,
            sample_time=sample_time,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
        )
        self._stiction = stiction
        super().__init__(
            plant, setpoint=setpoint, max_steps=max_steps,
            action_low=action_low, action_high=action_high,
            render_mode=render_mode,
        )
        high = np.array([1e4, 1e4], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._hist_vel: deque = deque(maxlen=self._RENDER_WINDOW)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._hist_vel.clear()
        vel = getattr(self.plant, "velocity", 0.0)
        return np.array([self.plant.output, vel], dtype=np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        vel = getattr(self.plant, "velocity", 0.0)
        self._hist_vel.append(vel)
        return np.array([obs[0], vel], dtype=np.float32), reward, terminated, truncated, info

    def _init_render(self):
        import matplotlib
        if _tk_available():
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        plt.ion()
        self._fig, (self._ax_out, self._ax_vel, self._ax_ctrl) = plt.subplots(
            3, 1, figsize=(9, 8), sharex=True
        )
        self._fig.suptitle("FrictionPlantEnv  —  live render", fontsize=12)

        self._ax_out.set_ylabel("Position")
        self._ax_out.grid(True, alpha=0.3)
        self._ax_vel.set_ylabel("Velocity")
        self._ax_vel.axhline(0, color="gray", lw=0.8, ls="--")
        self._ax_vel.grid(True, alpha=0.3)
        self._ax_ctrl.set_ylabel("Force (control)")
        self._ax_ctrl.set_xlabel("Time (s)")
        self._ax_ctrl.axhline(self._stiction, color="orange", lw=1.0, ls=":", label="±stiction")
        self._ax_ctrl.axhline(-self._stiction, color="orange", lw=1.0, ls=":")
        self._ax_ctrl.legend(fontsize=8)
        self._ax_ctrl.grid(True, alpha=0.3)

        (self._line_obs,) = self._ax_out.plot([], [], "b-", lw=1.5, label="Position")
        (self._line_sp,) = self._ax_out.plot([], [], "r--", lw=1.5, label="Setpoint")
        self._ax_out.legend(loc="upper left", fontsize=8)
        (self._line_vel,) = self._ax_vel.plot([], [], "c-", lw=1.2)
        (self._line_ctrl,) = self._ax_ctrl.plot([], [], "g-", lw=1.2)

        plt.tight_layout()
        self._fig.canvas.draw()

    def _update_render(self):
        import matplotlib.pyplot as plt
        if self._fig is None:
            return
        t = list(self._hist_t)
        if not t:
            return

        self._line_obs.set_data(t, list(self._hist_obs))
        self._line_sp.set_data(t, list(self._hist_sp))
        vel_data = list(self._hist_vel)
        if vel_data:
            self._line_vel.set_data(t[-len(vel_data):], vel_data)
        self._line_ctrl.set_data(t, list(self._hist_ctrl))

        for ax in (self._ax_out, self._ax_vel, self._ax_ctrl):
            ax.relim()
            ax.autoscale_view()

        self._fig.canvas.flush_events()
        plt.pause(0.001)

    # _ax_err alias not used in FrictionPlantEnv — override to avoid AttributeError
    def _get_info(self):
        return {
            "time": self.plant.time,
            "output": self.plant.output,
            "velocity": getattr(self.plant, "velocity", 0.0),
            "step": self._step_count,
        }


class FOPDTEnv(PIDPlantEnv):
    """Gymnasium env wrapping :class:`FOPDTPlant`."""

    def __init__(
        self,
        gain: float = 1.0,
        time_constant: float = 1.0,
        dead_time: float = 0.5,
        sample_time: float = 0.01,
        initial_output: float = 0.0,
        setpoint: float = 0.0,
        max_steps: int = 1000,
        action_low: float = -100.0,
        action_high: float = 100.0,
        render_mode: Optional[str] = None,
    ):
        plant = FOPDTPlant(
            gain=gain,
            time_constant=time_constant,
            dead_time=dead_time,
            sample_time=sample_time,
            initial_output=initial_output,
        )
        super().__init__(
            plant, setpoint=setpoint, max_steps=max_steps,
            action_low=action_low, action_high=action_high,
            render_mode=render_mode,
        )


# ======================================================================
# Double Pendulum environment
# ======================================================================

class DoublePendulumEnv(gym.Env):
    """
    Gymnasium env wrapping :class:`DoublePendulumCart`.

    Observation (6,): [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
    Action (1,):      force applied to cart
    Reward:           negative sum of absolute angles (stabilisation goal)

    Render (render_mode="human")
    ----------------------------
    Live 2-panel figure:
      Left:  2-D cart-pendulum schematic that animates in real time.
      Right: time-series of both angles and cart position.

    Physics options
    ---------------
    ``integrator`` kwarg selects "rk4" (default) or "solve_ivp".
    ``friction`` sets cart rail friction.
    ``apply_impulse(magnitude)`` injects a one-step force disturbance.
    """

    metadata = {"render_modes": ["human"]}
    _RENDER_WINDOW = 400

    def __init__(
        self,
        cart_mass: float = 1.0,
        pendulum1_mass: float = 0.1,
        pendulum2_mass: float = 0.1,
        pendulum1_length: float = 0.5,
        pendulum2_length: float = 0.5,
        friction: float = 0.1,
        sample_time: float = 0.005,
        initial_angle1: float = 0.1,
        initial_angle2: float = 0.1,
        control_mode: str = "position",
        integrator: str = "rk4",
        max_steps: int = 2000,
        action_low: float = -150.0,
        action_high: float = 150.0,
        angle_limit: float = 1.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self._l1 = pendulum1_length
        self._l2 = pendulum2_length
        self.plant = DoublePendulumCart(
            cart_mass=cart_mass,
            pendulum1_mass=pendulum1_mass,
            pendulum2_mass=pendulum2_mass,
            pendulum1_length=pendulum1_length,
            pendulum2_length=pendulum2_length,
            friction=friction,
            sample_time=sample_time,
            initial_angle1=initial_angle1,
            initial_angle2=initial_angle2,
            control_mode=control_mode,
            integrator=integrator,
        )
        self.max_steps = max_steps
        self._step_count = 0
        self.angle_limit = angle_limit
        self.render_mode = render_mode

        high_obs = np.array([5, 10, np.pi, 50, np.pi, 50], dtype=np.float32)
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.float32(action_low),
            high=np.float32(action_high),
            shape=(1,),
            dtype=np.float32,
        )

        # Render state
        self._fig = None
        self._hist_t: deque = deque(maxlen=self._RENDER_WINDOW)
        self._hist_x: deque = deque(maxlen=self._RENDER_WINDOW)
        self._hist_th1: deque = deque(maxlen=self._RENDER_WINDOW)
        self._hist_th2: deque = deque(maxlen=self._RENDER_WINDOW)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.plant.reset()
        self._step_count = 0
        self._hist_t.clear()
        self._hist_x.clear()
        self._hist_th1.clear()
        self._hist_th2.clear()
        if self.render_mode == "human":
            self._init_render()
        return self._obs(), self._get_info()

    def step(self, action):
        control = float(np.clip(action, self.action_space.low, self.action_space.high)[0])
        self.plant.update(control)
        self._step_count += 1

        s = self.plant.state
        self._hist_t.append(self.plant.time)
        self._hist_x.append(s[0])
        self._hist_th1.append(np.degrees(s[2]))
        self._hist_th2.append(np.degrees(s[4]))

        obs = self._obs()
        theta1 = abs(s[2])
        theta2 = abs(s[4])
        reward = -(theta1 + theta2)

        terminated = theta1 > self.angle_limit or theta2 > self.angle_limit
        truncated = self._step_count >= self.max_steps

        if self.render_mode == "human":
            self._update_render()

        return obs, reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            if self._fig is None:
                self._init_render()
            self._update_render()

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None

    def apply_impulse(self, magnitude: float) -> None:
        """Inject a one-step force disturbance onto the cart."""
        self.plant.set_disturbance(magnitude)

    # ------------------------------------------------------------------
    def _obs(self):
        return self.plant.state.astype(np.float32)

    def _get_info(self):
        return {
            "time": self.plant.time,
            "cart_position": self.plant.cart_position,
            "theta1_deg": np.degrees(self.plant.pendulum1_angle),
            "theta2_deg": np.degrees(self.plant.pendulum2_angle),
            "stable": self.plant.is_stable(),
            "step": self._step_count,
        }

    def _init_render(self):
        import matplotlib
        if _tk_available():
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import FancyBboxPatch

        plt.ion()
        self._fig = plt.figure(figsize=(12, 7))
        self._fig.suptitle("DoublePendulumEnv  —  live render", fontsize=12)
        gs = gridspec.GridSpec(2, 2, figure=self._fig)

        # Left column: schematic (spans both rows)
        self._ax_anim = self._fig.add_subplot(gs[:, 0])
        margin = self._l1 + self._l2 + 0.3
        self._ax_anim.set_xlim(-margin, margin)
        self._ax_anim.set_ylim(-0.3, self._l1 + self._l2 + 0.3)
        self._ax_anim.set_aspect("equal")
        self._ax_anim.set_title("Pendulum schematic")
        self._ax_anim.set_xlabel("x (m)")
        self._ax_anim.set_ylabel("height (m)")
        self._ax_anim.grid(True, alpha=0.25)
        self._ax_anim.axhline(0, color="k", lw=1.2)

        cart_w, cart_h = 0.4, 0.15
        self._cart_patch = FancyBboxPatch(
            (-cart_w / 2, -cart_h), cart_w, cart_h,
            boxstyle="round,pad=0.02", linewidth=1.5,
            edgecolor="k", facecolor="steelblue", zorder=3
        )
        self._ax_anim.add_patch(self._cart_patch)
        (self._link1,) = self._ax_anim.plot([], [], "o-", color="darkorange",
                                              lw=3, ms=8, zorder=4, label="Link 1")
        (self._link2,) = self._ax_anim.plot([], [], "o-", color="crimson",
                                              lw=3, ms=8, zorder=4, label="Link 2")
        self._ax_anim.legend(loc="upper right", fontsize=8)

        # Top-right: angles
        self._ax_ang = self._fig.add_subplot(gs[0, 1])
        self._ax_ang.set_ylabel("Angle (deg)")
        self._ax_ang.axhline(0, color="gray", lw=0.8, ls="--")
        self._ax_ang.grid(True, alpha=0.3)
        (self._line_th1,) = self._ax_ang.plot([], [], color="darkorange", lw=1.5, label="θ₁")
        (self._line_th2,) = self._ax_ang.plot([], [], color="crimson", lw=1.5, label="θ₂")
        self._ax_ang.legend(fontsize=8)

        # Bottom-right: cart position
        self._ax_pos = self._fig.add_subplot(gs[1, 1])
        self._ax_pos.set_ylabel("Cart position (m)")
        self._ax_pos.set_xlabel("Time (s)")
        self._ax_pos.axhline(0, color="gray", lw=0.8, ls="--")
        self._ax_pos.grid(True, alpha=0.3)
        (self._line_x,) = self._ax_pos.plot([], [], "steelblue", lw=1.5)

        plt.tight_layout()
        self._fig.canvas.draw()

    def _update_render(self):
        import matplotlib.pyplot as plt
        if self._fig is None:
            return

        s = self.plant.state
        x = s[0]
        th1 = s[2]
        th2 = s[4]

        # Cart patch
        cart_w = 0.4
        self._cart_patch.set_x(x - cart_w / 2)

        # Link 1: pivot at cart top centre
        x1 = x + self._l1 * np.sin(th1)
        y1 = self._l1 * np.cos(th1)
        self._link1.set_data([x, x1], [0, y1])

        # Link 2: pivot at end of link 1
        x2 = x1 + self._l2 * np.sin(th2)
        y2 = y1 + self._l2 * np.cos(th2)
        self._link2.set_data([x1, x2], [y1, y2])

        # Follow cart
        margin = self._l1 + self._l2 + 0.3
        self._ax_anim.set_xlim(x - margin, x + margin)

        # Time-series
        t = list(self._hist_t)
        if t:
            self._line_th1.set_data(t, list(self._hist_th1))
            self._line_th2.set_data(t, list(self._hist_th2))
            self._line_x.set_data(t, list(self._hist_x))
            for ax in (self._ax_ang, self._ax_pos):
                ax.relim()
                ax.autoscale_view()

        self._fig.canvas.flush_events()
        plt.pause(0.001)
