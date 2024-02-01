# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:00:52 2023

@authors: 
    Matt Chandler (m.chander@bristol.ac.uk)
    Daniel Collins (daniel.collins@bristol.ac.uk)
    Robert Hughes (robert.hughes@bristol.ac.uk)
"""

from functools import partial
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sympy


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
    t : array_like of float
        Time, either a single value or multiple values in a 1D array (shape
        (M,)). 
    y_t : array_like of float
        State space vector(s) containing displacement in row 0, velocity in row
        1 (0th axis), evaluated at corresponding time in `t`. If multiple 
        vectors are given, these should be in the columns (1st axis) (shape
        (2,) or (2, M)).
    mass : float
        Mass of the system.
    f_load : callable
        Function which expresses the time-dependent external load applied to
        the system. Takes an array_like of float as its only argument,
        returning an array_like of float. Signature `f = f_load(t)`.
    f_spring : callable
        Function which expresses the restoring elastic force on the system.
        Takes an array_like of float as its only argument, returning an
        array_like of float. Signature `f = f_spring(x)`.
    f_damp : callable
        Function which expresses the resistive damping force present in the
        system. Takes an array_like of float as its only argument, returning an
        array_like of float. Signature `f = f_damp(dx)`.

    Returns
    -------
    dy : array_like of float
        Velocity and acceleration at the point(s) `y_t` in row 0 and 1
        respectively, evaluated at time points given in `t` (shape (2,) or
        (2, M)).

    """
    t = np.asarray(t).squeeze()
    y_t = np.asarray(y_t).squeeze()
    if t.ndim == 0:
        if y_t.ndim != 1 or y_t.size != 2:
            raise ValueError(f'`y_t` must have shape (2,), found {y_t.shape}.')
    elif t.ndim == 1:
        if y_t.ndim != 2 or y_t.shape[1] != t.size:
            raise ValueError(
                f'`y_t` must have shape (2, {t.size}) when `t` has shape {t.shape}, found {y_t.shape}.'
            )
    else:
        raise ValueError(
            f'`t` must either be a single value or a 1D vector, found shape {t.shape}.'
        )

    # State space vector
    dy = np.asarray([
        y_t[1],
        (f_load(t) - f_spring(y_t[0]) - f_damp(y_t[1])) / mass,
    ])

    return dy


# %% Forces.


class Force:
    """
    An object which represents a force requires a list of the variables which
    are required as input so that only the relevant ones are provided later;
    a callable, for which variables may be provided; and ideally a printable
    expression to make it easier to visualise the model.

    Attributes
    ----------
    args : set
        A complete set of unique variables which the force requires to evaluate
        the result. Each element is an instance of `sympy.core.symbol.Symbol`.
        It is recommended that the variable name and `sympy` symbol are
        identical to each other, and identical to the name of the argument in
        the corresponding callable in `fn`.
    expr : sympy.core.Expr
        A printable expression which may be used in addition to other forces to
        produce an overall expression for the model. Use of `sympy` aids this
        very well.
    fn : callable
        The function which may be evaluated to determine the value of the force
        for the set of input variables which are named in `args`. Note that at
        the time when an instance of the `Force` class is defined, no actual
        numbers are required. Instead, this represents the interaction between
        the input variables.
    
    Examples
    --------
    >>> from sympy.abc import alpha
    >>> b, c_10 = sympy.symbols('b c_10')
    >>> def arbitrary_force(alpha, b, c_10):
    ...     return alpha * b + c_10
    >>> arb = Force(
    ...     (alpha, b, c_10,),
    ...     alpha * b + c_10,
    ...     arbitrary_force,
    ... )
    >>> arb.fn(1, 2, 3)
    5

    """

    def __init__(self, args, expr, fn=None):
        """
        Initialises an instance of the `Force` class.

        Parameters
        ----------
        args : iter
            Contains the set of variables which are used as input to `fn`. Note
            that they do not have to be identical to everything contained in
            `expr`, but they must be a complete representation of the arguments
            required for `fn`.
        expr : sympy.core.Expr
            An expression which is used for printing the model in a nicely
            formatted way. Note that `fn` is optional: if `fn` is not provided,
            then the `expr.lambdify` method is used to determine the callable.
            This will typically only be possible for quite simple functions
            (i.e. when a simple `lambda` would typically suffice).
        fn : callable, optional
            Callable which is used to evaluate the value of the force. Not
            required, but it is recommended that it is provided as there are
            only a handful of instances when the function can be derived from
            `expr`. The default is None.

        """
        for arg in args:
            if not isinstance(arg, sympy.core.symbol.Symbol):
                raise ValueError(f'arg `{arg}` expected to be a sympy Symbol.')
        if not isinstance(expr, sympy.core.Expr):
            raise ValueError(f'expr `{expr}` expected to be a sympy Expression.')

        # `args` are used with `fn` later, can be different to `expr`.
        self.args = {f'{arg}' for arg in args}
        self.expr = expr
        if fn is None:
            self.fn = sympy.lambdify(args, expr, 'numpy')
        else:
            self.fn = fn


# List of all sympy variables to be used in the following force functions. Note
# that if any greek letters are required, these can be imported from `sympy.abc`
# at the beginning of this script, or defined as you would in LaTeX.
t, x, v, m, k, c, F_0, t_0, f_F0, g, mu, F_prop, v_r, F_f, f_0, f_d, T_d, c_cr, F_f, epsilon = sympy.symbols(
    't x v m k c F_0 t_0 f_F0 g mu F_prop v_r F_f f_0 f_d T_d c_cr F_f epsilon'
)
alpha, delta, omega_0, omega_d, pi, zeta_0 = sympy.symbols(
    '\\alpha \\delta \\omega_0 \\omega_d \\pi \\zeta_0'
)


# Time-dependent external forcing

# No load applied
free_vibration = Force(
    (t,),
    sympy.Integer(0),
    lambda t: 0 * t
)


def const_load_fn(t, F_0, t_0):
    """
    Turn the force on after some time, i.e. when Δt > 0.
    If `t` is a 1D ndarray then `.squeeze()` should do nothing; if `t` is a
    float then return an object with zero dimensions.
    """
    sign_t = np.vstack([0 * t, np.sign(t - t_0)]).max(axis=0).squeeze()
    return F_0 * sign_t


# Constant force applied after some time t0
const_load = Force(
    (t, F_0, t_0),
    F_0,
    const_load_fn,
)

# Sinusoidal force
sin_load = Force(
    (t, F_0, f_F0),
    F_0 * sympy.sin(2 * pi * f_F0 * t),
    lambda t, F_0, f_F0: F_0 * np.sin(2 * np.pi * f_F0 * t).squeeze(),
)


# Spring equations

# Equation for a linear spring
linear_spring = Force(
    (x, k),
    x * k,
)


# Damping equations

# Undamped
undamped = Force(
    (v,),
    sympy.Integer(0),
    lambda v: 0 * v
)

# Linear viscous damping
linear_damping = Force(
    (v, c),
    v * c,
)


def sign(v, tol=1e-3):
    """
    Helper function which operates in the same way as np.sign, but everything
    below some tolerance is considered equal to zero.
    """
    v_nonzero = np.abs(v) > tol
    if v_nonzero.all():
        s = np.sign(v)
    else:
        s = np.zeros(v.shape)
        np.sign(v, out=s, where=v_nonzero)
    return s

# Sign approximation for friction equation
sign_friction = Force(
    (v, m, g, mu),
    F_f * sympy.sign(v),
    lambda v, m, g, mu: sign(v) * np.abs(m * g * mu),
)

# Hyperbolic tangent approximation for friction
tanh_friction = Force(
    (v, F_prop, v_r, m, g, mu),
    F_f * sympy.tanh(alpha * v),
    lambda v, F_prop, v_r, m, g, mu: np.abs(m * g * mu)
        * np.tanh(v * np.arctanh(F_prop) / v_r),
)

# Square root approximation for friction
sqrt_friction = Force(
    (v, m, g, mu, epsilon),
    F_f * v / sympy.sqrt(v**2 + epsilon**2),
    lambda v, m, g, mu, epsilon: (
        np.abs(m * g * mu) * v / np.sqrt(v**2 + epsilon**2)
    ),
)


# %% Plotting

# Collections of forces to be included. If a new force is added, make sure to
# also add it to these dictionaries with an appropriate name.
load_dict = {
    'Free Vibration': free_vibration,
    'Constant Force': const_load,
    'Sinusoidal': sin_load,
}
spring_dict = {
    'Linear': linear_spring,
}
damping_dict = {
    'Undamped': undamped,
    'Linear Viscous': linear_damping,
    'Coulomb Friction (sign)': sign_friction,
    'Coulomb Friction (tanh)': tanh_friction,
    'Coulomb Friction (sqrt)': sqrt_friction,
}
# Parameters used when enabling/disabling sliders
load_params = {'F_0', 't_0', 'f_F0'}
spring_params = {'k'}
damping_params = {'c', 'mu', 'F_prop', 'v_r', 'epsilon'}


class VibSimulation:
    """
    Object which does the simulation of the vibrating system, and then produces
    plots to visualise the system.

    Attributes
    ----------
    params : dict
        Collection of all possible parameters (including the ones which may not
        not be used for the active model), typically scalar valued.
    load_list : ipywidgets.widgets.widgets.Dropdown
        Set of possible loading forces obtained from `load_dict`.
    spring_list : ipywidgets.widgets.widgets.Dropdown
        Set of possible spring forces obtained from `spring_dict`.
    damping_list : ipywidgets.widgets.widgets.Dropdown
        Set of possible damping forces obtained from `damping_dict`.
    load : Force
        The loading force which is currently active.
    spring : Force
        The spring force which is currently active.
    damping : Force
        The damping force which is currently active.
    expr : ipywidgets.widgets.widgets.HTMLMath
        The equation representing the model which has been simulated.
    sliders : dict[ipywidgets.widgets.widgets.FloatSlider]
        Sliders allowing each of the parameters to be adjusted. The keys for
        each slider are identical to the associated value in `params`.
    checkboxes : dict[ipywidgets.widgets.widgets.Checkbox]
        Checkboxes allowing the user to enable / disable a given plot, if only
        interested in a handful.
    plot_output : ipywidgets.widgets.widgets.Output
        Where the plots are stored.
    controls : ipywidgets.widgets.widgets.VBox
        Where each of the interactive elements are stored. Contains (`expr`,
        `load_list`, `spring_list`, `damping_list`, `*sliders`, `*checkboxes`)
    layout : ipywidgets.widgets.widgets.HBox
        The complete layout which is drawn in the notebook.

    Methods
    -------
    run_simulation():
        Simulates the system with the current values which have been entered.
    on_load_change(_):
        Updates `self.load` to contain the new loading force chosen by the user
        and updates the plots.
    on_spring_change(_):
        Updates `self.spring` to contain the new spring force chosen by the
        user and updates the plots.
    on_damping_change(_):
        Updates `self.damping` to contain the new damping force chosen by the
        user and updates the plots.
    write_equation():
        Updates the model equation based on the currently active forces.
    on_slider_change(_):
        When a slider is interacted with, extract the new value and store it in
        `self.params`.
    on_checkbox_change(_):
        When a checkbox is toggled, update the figures only drawing the ones
        which are desired.
    update_figs():
        Redraw all of the desired figures based on the current values of the
        parameters.
    """

    def __init__(self, N_sa=250, N_Td=20, solver='BDF'):
        """
        Produces a set of figures to adjust SHM behaviour, using HTML widgets.
        As such, it is required that this is run from a jupyter notebook.

        Parameters
        ----------
        N_sa : int, optional
            Number of samples per period to model. The default is 50.
        N_Td : int, optional
            Number of periods to integrate over. The default is 20.
        """
        wgt_width = '400px'
        desc_width = '120px'

        # Initialise force lists
        self.load_list = widgets.Dropdown(
            options     = tuple(load_dict.keys()),
            value       = list(load_dict.keys())[0],
            description = 'External force:',
            style       = {'description_width': desc_width},
            layout      = {'width': wgt_width},
        )
        self.load_list.observe(self.on_load_change, names='value')
        self.load = load_dict[self.load_list.value]

        self.spring_list = widgets.Dropdown(
            options     = tuple(spring_dict.keys()),
            value       = list(spring_dict.keys())[0],
            description = 'Spring force:',
            style       = {'description_width': desc_width},
            layout      = {'width': wgt_width},
        )
        self.spring_list.observe(self.on_spring_change, names='value')
        self.spring = spring_dict[self.spring_list.value]

        self.damping_list = widgets.Dropdown(
            options     = tuple(damping_dict.keys()),
            value       = list(damping_dict.keys())[1],
            description = 'Damping force:',
            style       = {'description_width': desc_width},
            layout      = {'width': wgt_width},
        )
        self.damping_list.observe(self.on_damping_change, names='value')
        self.damping = damping_dict[self.damping_list.value]

        # Initialise summary of properties
        self.expr = widgets.HTMLMath(
            value       = '',
            placeholder = 'expr',
            description = 'Model equation:',
            style       = {'description_width': desc_width},
            layout      = {'width': wgt_width},
        )

        # Initialise parameters and sliders
        self.params = {
            # Input parameters which also have sliders
            'x_0':     0.1,    # [m] initial displacement in t_init
            'v_0':     0.0,    # [m s^{-2}] initial velocity in t_init
            'm':       8.0,    # [kg] mass
            'k':     348.0,    # [kg s^{-2}] (linear) stiffness constant
            'c':       7.5,    # [kg s^{-1}] viscous damping constant
            'F_0':     1.0,    # [N] magnitude of constant force
            't_0':     1.0,    # [N] start time of constant force
            'f_F0':    1.0,    # [Hz] frequency of sinusoidal force
            'mu':      0.0019, # [-] coefficient of friction
            'F_prop':  0.99,   # [-] fraction of F_f when v = v_r
            'v_r':     0.01, # [m s^{-1}] velocity at which F_{D,Coulomb} = F_prop * F_f
            'epsilon': 0.01, # [m s^{-1}] velocity at which F_{D,Coulomb} = F_f / sqrt(2)
            
            # Parameters which are default valued.
            't_init':  0.,     # [s] initial time of integration
            'g':       9.81,   # [m s^{-2}] gravitational acceleration
            'N_sa': N_sa,      # [-] number of samples per damped period
            'N_Td': N_Td,      # [-] number of damped periods for integration
            'solver': solver   # [-] which solver to use for integration.
        }

        self.sliders = {
            'x_0': widgets.FloatSlider(
                value=self.params['x_0'],
                min=0.0,
                max=1.0,
                step=0.01,
                description='Init $x_0$ (m)',
                tooltip='Initial displacement',
                style={'description_width': desc_width},
                layout={'width': wgt_width},
                continuous_update=False,
            ),
            'v_0': widgets.FloatSlider(
                value=self.params['v_0'],
                min=0.0,
                max=5.0,
                step=0.1,
                description = 'Init $v_0$ (m s$^{-1}$)',
                tooltip='Initial velocity',
                style={'description_width': desc_width},
                layout={'width': wgt_width},
                continuous_update=False,
            ),
            'm': widgets.FloatSlider(
                value=self.params['m'],
                min=1.0,
                max=100.0,
                step=1.0,
                description='$m$ (kg)',
                tooltip='Mass',
                style={'description_width': desc_width},
                layout={'width': wgt_width},
                continuous_update=False,
            ),
            'k': widgets.FloatSlider(
                value=self.params['k'],
                min=100.0,
                max=5000.0,
                step=10.0,
                description='$k$ (kg s$^{-2}$)',
                tooltip='Linear stiffness constant',
                style={'description_width': desc_width},
                layout={'width': wgt_width},
                continuous_update=False,
            ),
            'c': widgets.FloatSlider(
                value=self.params['c'],
                min=0.0,
                max=50.0,
                step=0.01,
                description='$c$ (kg s$^{-1}$)',
                tooltip='Viscous damping constant',
                style={'description_width': desc_width},
                layout={'width': wgt_width},
                continuous_update=False,
            ),
            'F_0': widgets.FloatSlider(
                value=self.params['F_0'],
                min=0.1,
                max=10.0,
                step=0.1,
                description='$F_0$ (N)',
                tooltip='Magnitude of constant force',
                style={'handle_color': 'lightgray', 'description_width': desc_width},
                layout={'width': wgt_width},
                disabled=True,
                continuous_update=False,
            ),
            't_0': widgets.FloatSlider(
                value=self.params['t_0'],
                min=0.01,
                max=15.0,
                step=0.1,
                description='$t_0$ (s)',
                tooltip='Time at which constant force applied',
                style={'handle_color': 'lightgray', 'description_width': desc_width},
                layout={'width': wgt_width},
                disabled=True,
                continuous_update=False,
            ),
            'f_F0': widgets.FloatSlider(
                value=self.params['f_F0'],
                min=0.1,
                max=5.0,
                step=0.1,
                description='$f_{F0}$ (Hz)',
                tooltip='Frequency of sinusoidal force',
                style={'handle_color': 'lightgray', 'description_width': desc_width},
                layout={'width': wgt_width},
                disabled=True,
                continuous_update=False,
            ),
            'mu': widgets.FloatSlider(
                value=self.params['mu'],
                min=0.1,
                max=2.0,
                step=0.05,
                description='$\mu$',
                tooltip='Coefficient of friction',
                style={'handle_color': 'lightgray', 'description_width': desc_width},
                layout={'width': wgt_width},
                disabled=True,
                continuous_update=False,
            ),
            'F_prop': widgets.FloatSlider(
                value=self.params['F_prop'],
                min=0.01,
                max=1.0,
                step=0.01,
                description='$F_{prop}$',
                tooltip='Fraction of friction force when v = v_r',
                style={'handle_color': 'lightgray', 'description_width': desc_width},
                layout={'width': wgt_width},
                disabled=True,
                continuous_update=False,
            ),
            'v_r': widgets.FloatSlider(
                value=self.params['v_r'],
                min=0.0001,
                max=0.01,
                step=0.0001,
                description='$v_r$ (m s$^{-1}$)',
                tooltip='Velocity at which friction force is F_prop F_f',
                style={'handle_color': 'lightgray', 'description_width': desc_width},
                layout={'width': wgt_width},
                disabled=True,
                continuous_update=False,
                readout_format='.3e',
            ),
            'epsilon': widgets.FloatSlider(
                value=self.params['epsilon'],
                min=0.0001,
                max=0.001,
                step=0.0001,
                description='$\epsilon$ (m s$^{-1}$)',
                tooltip='Velocity at which friction force is F_f / sqrt(2)',
                style={'handle_color': 'lightgray', 'description_width': desc_width},
                layout={'width': wgt_width},
                disabled=True,
                continuous_update=False,
                readout_format='.3e',
            ),
        }
        [self.sliders[arg].observe(self.on_slider_change) for arg in self.sliders.keys()]

        self.checkboxes = {
            'dva': widgets.Checkbox(
                value=True,
                description='Velocity, acceleration response',
                style={'description_width': desc_width},
                layout={'width': '500px'},
            ),
            'forces': widgets.Checkbox(
                value=True,
                description='Forces in time domain',
                style={'description_width': desc_width},
                layout={'width': wgt_width},
            ),
            'state-space': widgets.Checkbox(
                value=False,
                description='State space response',
                style={'description_width': desc_width},
                layout={'width': wgt_width},
            ),
        }
        [self.checkboxes[arg].observe(self.on_checkbox_change) for arg in self.checkboxes.keys()]
        
        self.derived_params = widgets.HTMLMath(
            value       = '',
            placeholder = 'params',
            description = 'Derived parameters:',
            style       = {'description_width': desc_width},
            layout      = {'width': wgt_width},
        )

        self.plot_output = widgets.Output()
        self.controls = widgets.VBox([
            self.expr,
            self.load_list,
            self.spring_list,
            self.damping_list,
            *self.sliders.values(),
            *self.checkboxes.values(),
            self.derived_params,
        ])
        self.layout = widgets.HBox([self.plot_output, self.controls])

        # Draw the first frame
        display(self.layout)
        self.update_figs()

    def run_simulation(self):
        """Compute the solution for the current set of parameters."""
        self.f_load = partial(
            self.load.fn,
            **{key: val for key, val in self.params.items() if key in self.load.args},
        )
        self.f_spring = partial(
            self.spring.fn,
            **{key: val for key, val in self.params.items() if key in self.spring.args},
        )
        self.f_damping = partial(
            self.damping.fn,
            **{key: val for key, val in self.params.items() if key in self.damping.args},
        )

        t_period = 2 * np.pi * np.sqrt(self.params['m'] / self.params['k'])
        self.soln = solve_ivp(
            system_1dof,
            [self.params['t_init'], t_period * self.params['N_Td']],
            [self.params['x_0'], self.params['v_0']],
            args=(self.params['m'], self.f_load, self.f_spring, self.f_damping),
            t_eval=np.linspace(
                self.params['t_init'],
                t_period * self.params['N_Td'],
                round(
                    (t_period * self.params['N_Td'] - self.params['t_init'])
                    * self.params['N_sa']
                    / t_period
                ),
            ),
            rtol=1e-8,
            atol=1e-8,
            method=self.params['solver'],
        )

        ddy = system_1dof(
            self.soln.t,
            self.soln.y,
            self.params['m'],
            self.f_load,
            self.f_spring,
            self.f_damping,
        )[1]
        self.y = np.vstack([self.soln.y, ddy])

    def on_load_change(self, _):
        """Update the GUI based on the new loading force."""
        self.load = load_dict[self.load_list.value]

        # Update sliders
        for param in self.sliders.keys():
            if param in load_params:
                if param in self.load.args:
                    self.sliders[param].style.handle_color = 'white'
                    self.sliders[param].disabled = False
                else:
                    self.sliders[param].style.handle_color = 'lightgray'
                    self.sliders[param].disabled = True

        self.update_figs()

    def on_spring_change(self, _):
        """Update the GUI based on the new spring force."""
        self.spring = spring_dict[self.spring_list.value]

        # Update sliders
        for param in self.sliders.keys():
            if param in spring_params:
                if param in self.spring.args:
                    self.sliders[param].style.handle_color = 'white'
                    self.sliders[param].disabled = False
                else:
                    self.sliders[param].style.handle_color = 'lightgray'
                    self.sliders[param].disabled = True

        self.update_figs()

    def on_damping_change(self, _):
        """Update the GUI based on the new damping force."""
        self.damping = damping_dict[self.damping_list.value]

        # Update sliders
        for param in self.sliders.keys():
            if param in damping_params:
                if param in self.damping.args:
                    self.sliders[param].style.handle_color = 'white'
                    self.sliders[param].disabled = False
                else:
                    self.sliders[param].style.handle_color = 'lightgray'
                    self.sliders[param].disabled = True

        self.update_figs()

    def write_equation(self):
        """Update the model equation from the newly selected forces."""
        self.expr.value = '${}$'.format(
            sympy.printing.latex(
                sympy.Eq(self.load.expr, self.spring.expr + self.damping.expr)
            )
        )

    def on_slider_change(self, _):
        """Extract the new parameter values and update the GUI."""
        for arg in self.sliders.keys():
            self.params[arg] = self.sliders[arg].value

        self.update_figs()

    def on_checkbox_change(self, _):
        """Update the GUI based on the new selection of figures."""
        self.update_figs(run_sim=False)
        
    def update_derived_params(self):
        """Update the system parameters which are derived from inputs."""  
        derived_params = {}
        derived_params[omega_0] = np.sqrt(self.params['k'] / self.params['m'])
                # [rad s^{-1}] undamped natural angular frequency
        derived_params[f_0] = derived_params[omega_0] / (2 * np.pi)
                # [Hz] undamped natural frequency
                
        if self.damping_list.value != 'Undamped':
            derived_params[delta] = self.params['c'] / (2 * self.params['m'])
                    # [s^{-1}] rate of exponential decay / rise
            derived_params[zeta_0] = derived_params[delta] / derived_params[omega_0]
                    # [-] damping ratio
            if derived_params[zeta_0] < 1:
                derived_params[omega_d] = derived_params[omega_0] * np.sqrt(
                    1 - derived_params[zeta_0]**2
                )   # [rad s^{-1}] damped angular natural frequency
                derived_params[f_d] = derived_params[omega_d] / (2 * np.pi)
                    # [Hz] damped natural frequency
                derived_params[T_d] = 1/ derived_params[f_d]
                    # [s] period of free damped response
            else:
                derived_params[omega_d] = 0
                derived_params[f_d] = 0
                derived_params[T_d] = np.inf
        
            derived_params[c_cr] = 2 * np.sqrt(
                self.params['m'] * self.params['k']
            )       # [kg s^{-1}] critical damping constant
        
        if 'Friction' in self.damping_list.value:
            derived_params[F_f] = abs(
                self.params['m'] * self.params['g'] * self.params['mu']
            )       # [N] magnitude of friction force
        
        self.derived_params.value = '<br>'.join(
            '${} = {}$'.format(sympy.printing.latex(key), sympy.Float(value, 4)) for key, value in derived_params.items()
        )
        

    def update_figs(self, run_sim=True):
        """Redraw all of the selected figures based on the user input."""
        if run_sim:
            self.write_equation()
            self.update_derived_params()
            self.run_simulation()

        with self.plot_output:
            clear_output(wait=True)
            
            # Initialise figure.
            num_plots = 1
            hratios = [1]
            height = 3.2
            if self.checkboxes['dva'].value:
                num_plots += 2
                hratios.append(.5)
                hratios.append(.5)
                height += 3.2
            if self.checkboxes['forces'].value:
                num_plots += 1
                hratios.append(.5)
                height += 1.6
                
            fig1, axs = plt.subplots(
                num_plots,
                1,
                sharex=True,
                height_ratios=hratios,
                figsize=(6, height),
                dpi=100
            )
            plot_idx = 0
            if num_plots == 1:
                axs = [axs]
            
            # Displacement figure.
            axs[plot_idx].grid()
            axs[plot_idx].plot(
                self.soln.t,
                self.f_load(self.soln.t) / self.params['k'],
                'r--',
                linewidth=2,
                label='Static response',
            )
            axs[plot_idx].plot(
                self.soln.t,
                self.y[0, :],
                'k',
                linewidth=2,
                label='Dynamic response',
            )
            axs[plot_idx].scatter(
                self.params['t_init'],
                self.params['x_0'],
                s=100,
                c='w',
                marker='o',
                edgecolors='C2',
                linewidths=2,
            )
            axs[plot_idx].text(
                self.params['t_init'],
                self.params['x_0'],
                'IC: [$t_0$ = {:.2g}s, $x_0$ = {:.2g}m]'.format(
                    self.params['t_init'], self.params['x_0']
                ),
            )
            axs[plot_idx].set_ylabel('Displacement\n$x$ (m)')
            axs[plot_idx].set_title('Time domain response')
            axs[plot_idx].legend(loc='upper right')
            
            # Velocity and acceleration figure.
            if self.checkboxes['dva'].value:
                plot_idx += 1
                axs[plot_idx].grid()
                axs[plot_idx].plot(self.soln.t, self.y[1, :], 'k')
                axs[plot_idx].set_ylabel('Velocity\n$v$ (ms$^{-1}$)')
                plot_idx += 1
                axs[plot_idx].grid()
                axs[plot_idx].plot(self.soln.t, self.y[2, :], 'k')
                axs[plot_idx].set_ylabel('Acceleration\n$a$ (ms$^{-2}$)')
            
            # Component forces.
            if self.checkboxes['forces'].value:
                plot_idx += 1
                axs[plot_idx].grid()
                axs[plot_idx].plot(
                    self.soln.t,
                    self.f_load(self.soln.t),
                    'r',
                    label='External forcing',
                )
                axs[plot_idx].plot(
                    self.soln.t,
                    self.f_spring(self.y[0, :]),
                    'g', 
                    label='Spring force',
                )
                axs[plot_idx].plot(
                    self.soln.t,
                    self.f_damping(self.y[1, :]),
                    'b',
                    label='Damping force',
                )
                axs[plot_idx].set_ylabel('Force\n$F$ (N)')
                axs[plot_idx].legend(loc='upper right')
            
            axs[-1].set_xlabel('Time (s)')
            fig1.tight_layout()
            plt.show()

            # State space response.
            if self.checkboxes['state-space'].value:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
                ax.grid()
                ax.plot(self.y[0, :], self.y[1, :], 'k', linewidth=2)
                ax.scatter(
                    self.y[0, 0],
                    self.y[1, 0],
                    s=100,
                    c='w',
                    marker='o',
                    edgecolors='C2',
                    linewidths=2,
                )
                plt.text(
                    self.y[0, 0],
                    self.y[1, 0],
                    'IC: [$x_0$ = {:.2g}s, $v_0$ = {:.2g}m]'.format(
                        self.y[0, 0], self.y[1, 0]
                    ),
                    verticalalignment='bottom',
                    horizontalalignment='right',
                )
                ax.set_xlabel('Displacement (m)')
                ax.set_ylabel('Velocity (ms$^{-1}$)')
                ax.set_title('State space response')
                fig.tight_layout()
                plt.show()
            return


#%%
if __name__ == '__main__':
    # This will plot the default state - to use the interactive one, import
    # this in a Jupyter notebook.
    fig = VibSimulation()