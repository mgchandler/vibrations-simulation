import numpy as np


# %% System equation


def system_1dof(
    t,
    y_t,
    mass,
    f_load,
    f_spring,
    f_damp,
):
    """
    Definition of the vibrating system with 1 DOF.

    Parameters
    ----------
    t : float OR ndarray[float] (M,)
        Time, either a single value or multiple values in an array.
    y_t : ndarray[float] (2,) OR ndarray[float] (2, M)
        State space vector(s) containing displacement in row 0, velocity in row
        1, evaluated at corresponding time in `t`.
    mass : float
        Mass of the system.
    f_load : Callable
        Function which expresses the time-dependent external load applied to
        the system. Must have signature `f = f_load(t)`, where `t` and `f` both
        have type `float`.
    f_spring : Callable
        Function which expresses the restoring elastic force on the system.
        Must have signature `f = f_spring(x)`, where `x` and `f` both have type
        `float`.
    f_damp : Callable
        Function which expresses the resistive damping force present in the
        system. Must have signature `f = f_damp(dx)`, where `dx` and `f` both
        have type `float`.

    Returns
    -------
    dy : ndarray[float] (2,) OR ndarray[float] (2, M)
        Velocity and acceleration at the point(s) `y_t`.

    """
    t = np.asarray(t).squeeze()
    y_t = np.asarray(y_t).squeeze()
    if t.ndim == 0:
        if y_t.ndim != 1 or y_t.size != 2:
            raise ValueError(f"y_t must have shape (2,), found {y_t.shape}.")
    elif t.ndim == 1:
        if y_t.ndim != 2 or y_t.shape[1] != t.size:
            raise ValueError(
                f"y_t must have shape (2, {t.size}) when t has shape {t.shape}, found {y_t.shape}."
            )
    else:
        raise ValueError(
            f"t must either be a single value or a 1D vector, found shape {t.shape}."
        )

    # State space vector
    dy = np.asarray(
        [
            y_t[1],
            (f_load(t) - f_spring(y_t[0]) - f_damp(y_t[1])) / mass,
        ]
    )

    return dy


# %% Force equations

# Time-dependent external forcing


def f_free(t):
    """
    Free vibration.
    """
    # Multiply by `t` to ensure that the returned value has the same type.
    return 0 * t


def f_const(t, F_0, t_F0):
    """
    Constant force `F_0` applied after some time `t_F0`.
    """
    # Turn the force on after some time, i.e. when Δt > 0.
    # If `t` is a 1D ndarray then `.squeeze()` should do nothing; if `t` is a
    # float then return an object with zero dimensions.
    sign_t = np.vstack([0 * t, np.sign(t - t_F0)]).max(axis=0).squeeze()
    return F_0 * sign_t


def f_sin(t, F_0, f_F0):
    """
    Sinusoidal force `F_0 * sin(ωt)` with magnitude `F_0` and frequency `f_F0`.
    """
    sin_t = np.sin(2 * np.pi * f_F0 * t).squeeze()
    return F_0 * sin_t


# Spring equations


def f_spring(x, k):
    """
    Linear spring equation.
    """
    return x * k


# Damping equations


def f_visc(v, c):
    """
    Linear viscous damping equation.
    """
    return v * c


def f_friction_sign(v, F_f):
    """
    Sign approximation for friction equation.
    """
    return np.sign(v) * F_f


def f_friction_tanh(v, alpha, F_f):
    """
    Hyperbolic tangent approxomation for friction equation.
    """
    return np.tanh(alpha * v) * F_f


def f_friction_sqrt(v, F_f, eps=1e-4):
    """
    Sqrt approximation of sign for friction equation.
    """
    return F_f * v / np.sqrt(v**2 + eps**2)
