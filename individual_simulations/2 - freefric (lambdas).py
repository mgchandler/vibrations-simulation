# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:51:52 2023

@author: mc16535
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import vibrations as vib


# %% Input and system parameters

# Basic integration parameters
t_init = 0                             # [s] initial time of integration
N_Td = 7                               # [-] number of damped periods for integration
N_sa = 1e2                             # [-] number of samples per damped period

# Initial conditions
x_0 = 0.001                            # [m] initial displacement in t_init
dx_0 = 0                               # [m s^{-2}] initial velocity in t_init

# General parameters
g = 9.81                               # [m s^{-2}] gravitational acceleration

# Physical system parameters
m1 = 32                                # [kg] mass
k1 = 4000                              # [kg s^{-2}] (linear) stiffness constant


# Friction model parameters
mu = 0.0019                            # [-] coefficient of friction

F_reach = 0.99                         # [-] portion of F_f to be reached in V_reach
V_reach = 1e-4                         # [m s^{-1}] velocity to reach F_reach portion of F_f
alpha = np.arctanh(F_reach) / V_reach  # [s m^{-1}] scaling factor in tanh function

eps = 1e-4                             # [-] ?


# System vibration parameters
omega_0 = np.sqrt(k1 / m1)             # [rad s^{-1}] undamped angular natural frequency
f_0 = omega_0 / (2 * np.pi)            # [Hz] undamped natural frequency
T_0 = 1 / f_0                          # [s] undamped period
F_f = abs(m1 * g * mu)                 # [N] magnitude of friction force
dx_1T = 4 * F_f / k1                   # [m] period reduction in T_0

t_end = T_0 * N_Td                     # [s] end time of integration


# %% System integration

# Initial conditions
y_0 = [x_0, dx_0]                         # Initial state vector

# Forces acting on the mass
f_load = lambda t: 0 * t                  # Free vibration
f_spring = lambda x: k1 * x               # Linear spring: k*x
f_sign = lambda dx: np.sign(dx) * F_f     # Sign model: sgn(dx/dt) * F_f
f_tanh = (                                # Hyperbolic tangent model: tanh(alpha*dx/dt) * F_f
    lambda dx: np.tanh(alpha * dx) * F_f
)  
f_sqrt = (                                # Sqrt sgn approx model
    lambda dx: F_f * dx / np.sqrt(dx**2 + eps**2)
)

# Choose which damping function to use.
f_damp = f_sign


# Call the integrator
soln = solve_ivp(
    vib.system_1dof,
    [t_init, t_end],
    y_0,
    args=(m1, f_load, f_spring, f_damp),
    rtol=1e-8,
    atol=1e-8,
)


# %% Visualisation

t, y = soln.t, soln.y  # Results


# Time domain response - displacement
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
ax.grid()
ax.plot(t, y[0, :], "k", linewidth=2)
plt.scatter(t_init, x_0, s=100, c="w", marker="o", edgecolors="C2", linewidths=2)
plt.text(t_init, x_0, "IC: [$t_0$ = {:.2g}s, $x_0$ = {:.2g}m]".format(t_init, x_0))
ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement (m)")
ax.set_title("Time domain response of the 1 DOF system")
fig.tight_layout()
plt.show()


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
