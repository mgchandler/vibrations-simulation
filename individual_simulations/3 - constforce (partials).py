# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:51:52 2023

@author: mc16535
"""
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import vibrations_old as vib


# %% Input and system parameters

# Basic integration parameters
t_init = 0                         # [s] initial time of integration
t_end = 6                          # [s] end time of integration

# Initial conditions
x_0 = 0                            # [m] initial displacement in t_init
dx_0 = 0                           # [m s^{-2}] initial velocity in t_init

# Excitation parameters
F_0 = 1                            # [N] magnitude of constant force
t_F0 = 1                           # [N] start time of constant force

# Physical system parameters
m1 = 8                             # [kg] mass
k1 = 4000                          # [kg s^{-2}] (linear) stiffness constant
c1 = 11                            # [kg s^{-1}] viscous damping constant


# System vibration parameters
omega_0 = np.sqrt(k1 / m1)         # [rad s^{-1}] undamped angular natural frequency
f_nf0 = omega_0 / (2 * np.pi)      # [Hz] undamped natural frequency
delta = c1 / (2 * m1)              # [s^{-1}] rate of exponential decay / rise
zeta_0 = delta / omega_0           # [-] damping ratio

if zeta_0 < 1:
    omega_d = omega_0 * np.sqrt(   # [rad s^{-1}] damped angular natural frequency
        1 - zeta_0**2
    )  
    f_nfd = omega_d / (2 * np.pi)  # [Hz] damped natural frequency
    T_d = 1 / f_nfd                # [s] period of free damped response
else:
    omega_d = 0
    f_nfd = 0
    T_d = np.inf

c1cr = 2 * np.sqrt(m1 * k1)        # [kg s^{-1}] critical damping constant


# %% System integration

# Initial conditions
y_0 = [x_0, dx_0]                        # Initial state vector

# Forces acting on the mass. Use `functools.partial` to assign some (but not
# all) of the arguments.
f_load = partial(                        # External load: constant force after time delay
    vib.f_const, F_0=F_0, t_F0=t_F0
) 
f_spring = partial(vib.f_spring, k1=k1)  # Linear spring: k*x
f_damp = partial(vib.f_visc, c1=c1)      # Linear viscous damper: c * dx/dt


# Call the integrator
soln = solve_ivp(
    vib.system_1dof,
    [t_init, t_end],
    y_0,
    args=(m1, f_load, f_spring, f_damp),
    rtol=1e-12,
    atol=1e-12,
)


# %% Visualisation

t, y = soln.t, soln.y  # Results


# Time domain response - displacement
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
ax.grid()
ax.plot(t, y[0, :], "k", linewidth=2, label="Dynamic response")
ax.plot(t, f_load(t) / k1, "r--", linewidth=2, label="Static response")
plt.scatter(t_init, x_0, s=100, c="w", marker="o", edgecolors="C2", linewidths=2)
plt.text(t_init, x_0, "IC: [$t_0$ = {:.2g}s, $x_0$ = {:.2g}m]".format(t_init, x_0))
ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement (m)")
ax.set_title("Time domain response of the 1 DOF system")
ax.legend()
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
