# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:51:52 2023

@author: mc16535
"""
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import vibrations as vib

from dataclasses import dataclass, field, asdict
# import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

## freevisc
# freevisc dataclass
@dataclass
class FreeviscSystem:
    """Dataclass containing the freevisc system parameters and solution"""
    # Basic integration parameters
    t_init: float = 0
    N_Td: int = 15
    N_sa: float = 1e2

    # Initial conditions
    x_0: float = 0.1 # slider variable - freevisc
    dx_0: float = 0  # slider variable - freevisc

    # Physical system parameters
    m1: float = 16 # slider variable - freevisc
    k1: float = 348 # slider variable - freevisc
    c1: float = None # slider variable - freevisc

    # Computed System Vibration Parameters
    omega_0: float = field(init=False)
    f_0: float = field(init=False)
    delta: float = field(init=False)
    zeta_0: float = field(init=False)
    omega_d: float = field(init=False)
    f_d: float = field(init=False)
    T_d: float = field(init=False)
    t_end: float = field(init=False)
    t_eval: np.ndarray = field(init=False)
    c_cr: float = field(init=False)

    # Solution
    soln: object = field(init=False)
    y: np.ndarray = field(init=False)

    # Forces
    f_load: callable = field(init=False)
    f_spring: callable = field(init=False)
    f_damp: callable = field(init=False)

    def __post_init__(self):
        if self.c1 is None:
            self.c1 = 0.0501 * 2 * np.sqrt(self.m1 * self.k1)

        # Compute dependent parameters
        self.compute_dependent_parameters()

        # Set up forces
        self.f_load = vib.f_free
        self.f_spring = partial(vib.f_spring, k=self.k1)
        self.f_damp = partial(vib.f_visc, c=self.c1)

        # Run the simulation
        self.run_simulation()

    def compute_dependent_parameters(self):
        # Basic parameters
        self.omega_0 = np.sqrt(self.k1 / self.m1)          # Undamped angular natural frequency
        self.f_0 = self.omega_0 / (2 * np.pi)              # Undamped natural frequency
        self.delta = self.c1 / (2 * self.m1)               # Rate of exponential decay/rise
        self.zeta_0 = self.delta / self.omega_0            # Damping ratio
        self.c_cr = 2 * np.sqrt(self.m1 * self.k1)         # Critical damping constant

        # Underdamped case
        if self.zeta_0 < 1:
            self.omega_d = self.omega_0 * np.sqrt(1 - self.zeta_0**2)  # Damped angular natural frequency
            self.f_d = self.omega_d / (2 * np.pi)                      # Damped natural frequency
            self.T_d = 1 / self.f_d                                    # Period of free damped response
            self.t_end = self.T_d * self.N_Td                          # End time of integration
            self.t_eval = np.linspace(self.t_init, self.t_end, round((self.t_end - self.t_init) / self.T_d * self.N_sa))

        # Overdamped or critically damped case
        else:
            self.omega_d = 0
            self.f_d = 0
            self.T_d = np.inf
            self.t_end = 2 * self.N_Td / self.delta
            self.t_eval = None


    def run_simulation(self):
        y_0 = [self.x_0, self.dx_0] # 
        self.soln = solve_ivp(
            vib.system_1dof,
            [self.t_init, self.t_end],
            y_0,
            t_eval=self.t_eval,
            args=(self.m1, self.f_load, self.f_spring, self.f_damp),
            rtol=1e-12,
            atol=1e-12,
        )
        self.y = self.soln.y

# freevisc plotter
class FreeviscPlotter:
    def __init__(self):
        # plotting syles, titles, modes etc. 
        self.placeholder = None

    def plot_time_domain_response(self, t, y, t_init, x_0, y_0, zeta_0, delta, omega_d):
        # Time domain response - displacement
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        ax.grid()
        ax.plot(t, y[0, :], "k", linewidth=2)
        plt.scatter(t_init, x_0, s=100, c="w", marker="o", edgecolors="C2", linewidths=2)
        plt.text(t_init, x_0, "IC: [$t_0$ = {:.2g}s, $x_0$ = {:.2g}m]".format(t_init, x_0))
        if zeta_0 < 1:  # Exponential decay for undamped systems
            y_damp = np.sqrt(y_0[0] ** 2 + ((y_0[0] * delta + y_0[1]) / omega_d) ** 2)
            y_delta = y_damp * np.exp(-delta * t)
            ax.plot(t, y_delta, "r--", linewidth=2)
            ax.plot(t, -y_delta, "r--", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (m)")
        ax.set_title("Time domain response of the 1 DOF system")
        fig.tight_layout()
        plt.show()

    def plot_displacement_velocity_acceleration(self, t, y, m1, f_load, f_spring, f_damp):
        # Time domain response - disp, vel, acc
        ddy = vib.system_1dof(t, y, m1, f_load, f_spring, f_damp)[1]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 6), dpi=100)
        [axs[i].grid() for i in range(axs.size)]
        axs[0].plot(t, y[0, :], "k")
        axs[0].set_ylabel("$x$ (m)")
        axs[1].plot(t, y[1, :], "k")
        axs[1].set_ylabel("$\\frac{dx}{dt}$ (ms$^{-1}$)")
        axs[2].plot(t, ddy, "k")
        axs[2].set_ylabel("$\\frac{d^2x}{dt^2}$ (ms$^{-2}$)")
        axs[2].set_xlabel("Time (s)")
        fig.suptitle("Displacement, velocity and acceleration responses")
        fig.tight_layout()
        plt.show()

    def plot_state_space_response(self, y):
        # State space response
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        ax.grid()
        ax.plot(y[0, :], y[1, :], "k", linewidth=2)
        ax.scatter(y[0, 0], y[1, 0], s=100, c="w", marker="o", edgecolors="C2", linewidths=2)
        plt.text(
            y[0, 0],
            y[1, 0],
            "IC: [$x_0$ = {:.2g}s, $v_0$ = {:.2g}m]".format(y[0, 0], y[1, 0]),
            verticalalignment="bottom",
            horizontalalignment="right",
        )
        ax.set_xlabel("Displacement (m)")
        ax.set_ylabel("Velocity (ms$^{-1}$)")
        ax.set_title("Response of the 1 DOF system in the state space")
        fig.tight_layout()
        plt.show()

    def plot_forces_in_time_domain(self, t, y, f_load, f_spring, f_damp):
        # Forces in the time domain
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        ax.grid()
        ax.plot(t, f_load(t), "r", label="External forcing")
        ax.plot(t, f_spring(y[0, :]), "g", label="Spring force")
        ax.plot(t, f_damp(y[1, :]), "b", label="Damping force")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N)")
        ax.legend()
        fig.tight_layout()
        plt.show()

    def plot_component_forces(self, y, f_spring, f_damp):
        # Component forces
        fig, axs = plt.subplots(2, 2, figsize=(6, 4), dpi=100)
        [axs.ravel()[i].grid() for i in range(axs.size)]
        axs[0, 0].plot(y[0, :], f_spring(y[0, :]), "k", linewidth=2)
        axs[0, 0].set_ylabel("Spring force (N)")
        axs[0, 1].plot(y[1, :], f_spring(y[0, :]), "k", linewidth=2)
        axs[1, 0].plot(y[0, :], f_damp(y[1, :]), "k", linewidth=2)
        axs[1, 0].set_xlabel("Displacement (m)")
        axs[1, 0].set_ylabel("Dashpot force (N)")
        axs[1, 1].plot(y[1, :], f_damp(y[1, :]), "k", linewidth=2)
        axs[1, 1].set_xlabel("Velocity (ms$^{-1}$)")
        fig.tight_layout()
        plt.show()

## freefric
@dataclass
class FreefricSystem:
    """Dataclass containing the freefric system parameters and solution"""
    # Basic integration parameters
    t_init: float = 0
    N_td: int = 7
    N_sa: float = 1e2

    # Initial conditions
    x_0: float = 0.001 # slider variable - freefric
    dx_0: float = 0  # slider variable - freefric

    # General parameters
    g: float = 9.81 # slider variable - freefric

    # Physical system parameters
    m1: float = 8 # slider variable - freefric
    k1: float = 4000 # slider variable - freefric

    # Friction model parameters
    mu: float = 0.0019 # slider variable - freefric
    F_reach: float = 0.99
    V_reach: float = 1e-4
    alpha: float = field(init=False)
    eps: float = 1e-4

    # Computed System Vibration Parameters
    omega_0: float = field(init=False)
    f_nf0: float = field(init=False)
    T_0: float = field(init=False)
    F_f: float = field(init=False)
    dx_1T: float = field(init=False)
    t_end: float = field(init=False)

    # Solution
    soln: object = field(init=False)
    y: np.ndarray = field(init=False)

    # Forces
    f_load: callable = field(init=False)
    f_spring: callable = field(init=False)
    f_damp: callable = field(init=False)

    def __post_init__(self):
        # Compute dependent parameters
        self.compute_dependent_parameters()

        # Set up forces
        self.f_load = vib.f_free
        self.f_spring = partial(vib.f_spring, k=self.k1)
        self.f_damp = partial(vib.f_friction_sign, F_f=self.F_f)

        # Run the simulation
        self.run_simulation()

    def compute_dependent_parameters(self):
        # Compute dependent parameters
        self.omega_0 = np.sqrt(self.k1 / self.m1)          # Undamped angular natural frequency
        self.f_nf0 = self.omega
        self.T_0 = 1 / self.f_nf0                        # Undamped period
        self.F_f = abs(self.m1 * self.g * self.mu)                 # Magnitude of friction force
        self.dx_1T = 4 * self.F_f / self.k1                   # Period reduction in T_0
        self.t_end = self.T_0 * self.N_td                     # End time of integration
        self.alpha = np.arctanh(self.F_reach) / self.V_reach

    def run_simulation(self):
        y_0 = [self.x_0, self.dx_0]
        self.soln = solve_ivp(
            vib.system_1dof,
            [self.t_init, self.t_end],
            y_0,
            args=(self.m1, self.f_load, self.f_spring, self.f_damp),
            rtol=1e-8,
            atol=1e-8,
        )
        self.y = self.soln.y


# freefric plotter
class FreefricPlotter:
    def __init__(self):
        # plotting syles, titles, modes etc. 
        self.placeholder = None

    def plot_time_domain_response(self, t, y, t_init, x_0, y_0, zeta_0, delta, omega_d):
        # Time domain response - displacement
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        ax.grid()
        ax.plot(t, y[0, :], "k", linewidth=2)
        plt.scatter(t_init, x_0, s=100, c="w", marker="o", edgecolors="C2", linewidths=2)
        plt.text(t_init, x_0, "IC: [$t_0$ = {:.2g}s, $x_0$ = {:.2g}m]".format(t_init, x_0))
        if zeta_0 < 1:  # Exponential decay for undamped systems
            y_damp = np.sqrt(y_0[0] ** 2 + ((y_0[0] * delta + y_0[1]) / omega_d) ** 2)
            y_delta = y_damp * np.exp(-delta * t)
            ax.plot(t, y_delta, "r--", linewidth=2)
            ax.plot(t, -y_delta, "r--", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (m)")
        ax.set_title("Time domain response of the 1 DOF system")
        fig.tight_layout()
        plt.show()

    def plot_displacement_velocity_acceleration(self, t, y, m1, f_load, f_spring, f_damp):
        # Time domain response - disp, vel, acc
        ddy = vib.system_1dof(t, y, m1, f_load, f_spring, f_damp)[1]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 6), dpi=100)
        [axs[i].grid() for i in range(axs.size)]
        axs[0].plot(t, y[0, :], "k")
        axs[0].set_ylabel("$x$ (m)")
        axs[1].plot(t, y[1, :], "k")
        axs[1].set_ylabel("$\\frac{dx}{dt}$ (ms$^{-1}$)")
        axs[2].plot(t, ddy, "k")
        axs[2].set_ylabel("$\\frac{d^2x}{dt^2}$ (ms$^{-2}$)")
        axs[2].set_xlabel("Time (s)")
        fig.suptitle("Displacement, velocity and acceleration responses")
        fig.tight_layout()
        plt.show()

    def plot_state_space_response(self, y):
        # State space response
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        ax.grid()
        ax.plot(y[0, :], y[1, :], "k", linewidth=2)
        ax.scatter(y[0, 0], y[1, 0], s=100, c="w", marker="o", edgecolors="C2", linewidths=2)
        plt.text(
            y[0, 0],
            y[1, 0],
            "IC: [$x_0$ = {:.2g}s, $v_0$ = {:.2g}m]".format(y[0, 0], y[1, 0]),
            verticalalignment="bottom",
            horizontalalignment="right",
        )
        ax.set_xlabel("Displacement (m)")
        ax.set_ylabel("Velocity (ms$^{-1}$)")
        ax.set_title("Response of the 1 DOF system in the state space")
        fig.tight_layout()
        plt.show()

    def plot_forces_in_time_domain(self, t, y, f_load, f_spring, f_damp):
        # Forces in the time domain
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        ax.grid()
        ax.plot(t, f_load(t), "r", label="External forcing")
        ax.plot(t, f_spring(y[0, :]), "g", label="Spring force")
        ax.plot(t, f_damp(y[1, :]), "b", label="Damping force")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N)")
        ax.legend()
        fig.tight_layout()
        plt.show()
    
    def plot_component_forces(self, y, f_spring, f_damp):
        # Component forces
        fig, axs = plt.subplots(2, 2, figsize=(6, 4), dpi=100)
        [axs.ravel()[i].grid() for i in range(axs.size)]
        axs[0, 0].plot(y[0, :], f_spring(y[0, :]), "k", linewidth=2)
        axs[0, 0].set_ylabel("Spring force (N)")
        axs[0, 1].plot(y[1, :], f_spring(y[0, :]), "k", linewidth=2)
        axs[1, 0].plot(y[0, :], f_damp(y[1, :]), "k", linewidth=2)
        axs[1, 0].set_xlabel("Displacement (m)")
        axs[1, 0].set_ylabel("Dashpot force (N)")
        axs[1, 1].plot(y[1, :], f_damp(y[1, :]), "k", linewidth=2)
        axs[1, 1].set_xlabel("Velocity (ms$^{-1}$)")
        fig.tight_layout()
        plt.show()


##constforce
@dataclass
class ConstforceSystem:
    """Dataclass containing the constforce system parameters and solution"""
    # Basic integration parameters
    t_init: float = 0
    t_end: float = 6

    # Initial conditions
    x_0: float = 0 # slider variable - constforce
    dx_0: float = 0

    # Excitation parameters
    F_0: float = 1
    t_F0: float = 1

    # Physical system parameters
    m1: float = 8
    k1: float = 4000
    c1: float = 11

    # Computed System Vibration Parameters
    omega_0: float = field(init=False)
    f_nf0: float = field(init=False)
    delta: float = field(init=False)
    zeta_0: float = field(init=False)
    omega_d: float = field(init=False)
    f_nfd: float = field(init=False)
    T_d: float = field(init=False)
    c1cr: float = field(init=False)

    # Solution
    soln: object = field(init=False)
    y: np.ndarray = field(init=False)

    # Forces
    f_load: callable = field(init=False)
    f_spring: callable = field(init=False)
    f_damp: callable = field(init=False)

    def __post_init__(self):
        # Compute dependent parameters
        self.compute_dependent_parameters()

        # Set up forces
        self.f_load = partial(
            vib.f_const, F_0=self.F_0, t_F0=self.t_F0
        ) 
        self.f_spring = partial(vib.f_spring, k1=self.k1)
        self.f_damp = partial(vib.f_visc, c1=self.c1)

        # Run the simulation
        self.run_simulation()

    def compute_dependent_parameters(self):
        # Compute dependent parameters
        self.omega_0 = np.sqrt(self.k1 / self.m1)         # Undamped angular natural frequency
        self.f_nf0 = self.omega_0 / (2 * np.pi)          # Undamped natural frequency
        self.delta = self.c1 / (2 * self.m1)              # Rate of exponential decay/rise
        self.zeta_0 = self.delta / self.omega_0            # Damping ratio

        # Underdamped case
        if self.zeta_0 < 1:
            self.omega_d = self.omega_0 * np.sqrt(1 - self.zeta_0**2)
            self.f_nfd = self.omega_d / (2 * np.pi)
            self.T_d = 1 / self.f_nfd

        # Overdamped or critically damped case
        else:
            self.omega_d = 0
            self.f_nfd = 0
            self.T_d = np.inf

        self.c1cr = 2 * np.sqrt(self.m1 * self.k1) # Critical damping constant

    def run_simulation(self):
        y_0 = [self.x_0, self.dx_0]
        self.soln = solve_ivp(
            vib.system_1dof,
            [self.t_init, self.t_end],
            y_0,
            args=(self.m1, self.f_load, self.f_spring, self.f_damp),
            rtol=1e-12,
            atol=1e-12,
        )
        self.y = self.soln.y



# constforce plotter
class ConstforcePlotter:
    """Plotting class for the constforce system"""
    def __init__(self):
        # plotting syles, titles, modes etc. 
        self.placeholder = None

    def plot_time_domain_response(self, t, y, t_init, x_0, y_0, zeta_0, delta, omega_d):
        # Time domain response - displacement
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        ax.grid()
        ax.plot(t, y[0, :], "k", linewidth=2)
        plt.scatter(t_init, x_0, s=100, c="w", marker="o", edgecolors="C2", linewidths=2)
        plt.text(t_init, x_0, "IC: [$t_0$ = {:.2g}s, $x_0$ = {:.2g}m]".format(t_init, x_0))
        if zeta_0 < 1:  # Exponential decay for undamped systems
            y_damp = np.sqrt(y_0[0] ** 2 + ((y_0[0] * delta + y_0[1]) / omega_d) ** 2)
            y_delta = y_damp * np.exp(-delta * t)
            ax.plot(t, y_delta, "r--", linewidth=2)
            ax.plot(t, -y_delta, "r--", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (m)")
        ax.set_title("Time domain response of the 1 DOF system")
        fig.tight_layout()
        plt.show()

    def plot_displacement_velocity_acceleration(self, t, y, m1, f_load, f_spring, f_damp):
        # Time domain response - disp, vel, acc
        ddy = vib.system_1dof(t, y, m1, f_load, f_spring, f_damp)[1]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 6), dpi=100)
        [axs[i].grid() for i in range(axs.size)]
        axs[0].plot(t, y[0, :], "k")
        axs[0].set_ylabel("$x$ (m)")
        axs[1].plot(t, y[1, :], "k")
        axs[1].set_ylabel("$\\frac{dx}{dt}$ (ms$^{-1}$)")
        axs[2].plot(t, ddy, "k")
        axs[2].set_ylabel("$\\frac{d^2x}{dt^2}$ (ms$^{-2}$)")
        axs[2].set_xlabel("Time (s)")
        fig.suptitle("Displacement, velocity and acceleration responses")
        fig.tight_layout()
        plt.show()

    def plot_forces_in_time_domain(self, t, y, f_load, f_spring, f_damp):
        # Forces in the time domain
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        ax.grid()
        ax.plot(t, f_load(t), "r", label="External forcing")
        ax.plot(t, f_spring(y[0, :]), "g", label="Spring force")
        ax.plot(t, f_damp(y[1, :]), "b", label="Damping force")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N)")
        ax.legend()
        fig.tight_layout()
        plt.show()

# widgets
def update_plots(x_0, dx_0, m1, k1, c1):
    with plot_output:
        clear_output(wait=True) # Clear the previous output before displaying and updating plots

        params = FreeviscSystem(x_0=x_0, dx_0=dx_0, m1=m1, k1=k1, c1=c1)

        # Get parameters for plotting
        t = params.soln.t
        y = params.y
        t_init = params.t_init
        x_0 = params.x_0
        zeta_0 = params.zeta_0
        delta = params.delta
        omega_d = params.omega_d
        y_0 = [x_0, params.dx_0]  # Initial state vector

        f_damp = params.f_damp
        f_spring = params.f_spring
        f_load = params.f_load
        
        # Create an instance of the plotting class
        plotter = FreeviscPlotter()

        # Call plotting functions with extracted parameters based on checkbox states
        if time_domain_response_checkbox.value:
            plotter.plot_time_domain_response(t, y, t_init, x_0, y_0, zeta_0, delta, omega_d)
        if displacement_velocity_acceleration_checkbox.value:
            plotter.plot_displacement_velocity_acceleration(t, y, m1, f_load, f_spring, f_damp)
        if state_space_response_checkbox.value:
            plotter.plot_state_space_response(y)
        if forces_in_time_domain_checkbox.value:
            plotter.plot_forces_in_time_domain(t, y, f_load, f_spring, f_damp)
        if component_forces_checkbox.value:
            plotter.plot_component_forces(y, f_spring, f_damp)


# create checkbox for each plot to be displayed
time_domain_response_checkbox = widgets.Checkbox(value=True, description='Time Domain Response')
displacement_velocity_acceleration_checkbox = widgets.Checkbox(value=True, description='Displacement, Velocity, Acceleration')
state_space_response_checkbox = widgets.Checkbox(value=True, description='State Space Response')
forces_in_time_domain_checkbox = widgets.Checkbox(value=True, description='Forces in Time Domain')
component_forces_checkbox = widgets.Checkbox(value=True, description='Component Forces')

# Callback function for slider changes
def on_slider_change(change):
    update_plots(x_0_slider.value, dx_0_slider.value, m1_slider.value, k1_slider.value, c1_slider.value)

# Create sliders
x_0_slider = widgets.FloatSlider(min=0, max=1, step=0.01, value=0.1, description='Initial Displacement (m):', continuous_update=False)
dx_0_slider = widgets.FloatSlider(min=-5, max=5, step=0.1, value=0.0, description='Initial Velocity (m/s):', continuous_update=False)
m1_slider = widgets.FloatSlider(min=1, max=100, step=1, value=16, description='Mass (kg):', continuous_update=False)
k1_slider = widgets.FloatSlider(min=10, max=1000, step=10, value=348, description='Stiffness (kg/s^2):', continuous_update=False)
c1_slider = widgets.FloatSlider(min=0.01, max=100, step=0.01, value=0.0501 * 2 * np.sqrt(16 * 348), description='Damping (kg/s):', continuous_update=False)

# Attach the callback function so plots update when controls are changed
x_0_slider.observe(on_slider_change, names='value')
dx_0_slider.observe(on_slider_change, names='value')
m1_slider.observe(on_slider_change, names='value')
k1_slider.observe(on_slider_change, names='value')
c1_slider.observe(on_slider_change, names='value')

time_domain_response_checkbox.observe(on_slider_change, names='value')
displacement_velocity_acceleration_checkbox.observe(on_slider_change, names='value')
state_space_response_checkbox.observe(on_slider_change, names='value')
forces_in_time_domain_checkbox.observe(on_slider_change, names='value')
component_forces_checkbox.observe(on_slider_change, names='value')


# Output widget for displaying plots
plot_output = widgets.Output()

# Arrange sliders and checkboxes vertically
controls = widgets.VBox([x_0_slider, dx_0_slider, m1_slider, k1_slider, c1_slider,
                         time_domain_response_checkbox, displacement_velocity_acceleration_checkbox,
                         state_space_response_checkbox, forces_in_time_domain_checkbox, component_forces_checkbox])

# Create a horizontal box to place sliders next to the plot output
layout = widgets.HBox([plot_output, controls])

# Display the layout
display(layout)

# Initialize the plots
update_plots(x_0_slider.value, dx_0_slider.value, m1_slider.value, k1_slider.value, c1_slider.value)
