# vibration-system

This repository contains code for the demonstration of physical vibration systems under various conditions. 

The home directory contains the four files written by X which are the basis of the implementation used in the Jupyter notebook.
* 1 - `freevisc.py`: This Python script contains variables and functions related to free vibrations with viscous damping.
* 2 - `freefric.py`: This Python script contains variables and functions related to free vibrations with friction.
* 3 - `constforce.py`: This Python script contains contains variables and functions related to vibrations under a constant force.
* 4 `vibrations.py`: This Python script contains the main functions and classes used in the simulation of the vibration system.


The files `freevisc.py`, `freefric.py`, `constforce.py` each include parameters and functions for solving the differential equations of motion for the vibration system under the given conditions, as well as methods for plotting the system. The file `vibrations.py` contains the `VibrationSystem` class, which is used to simulate the vibration system under the different conditions described by the other three files.

Each of the condition scripts defines various parameters related to the system, e.g. mass, stiffness constant, friction coefficient, and initial conditions. The scripts then define the forces acting on the system (free vibration force, a linear spring force, and a friction force) using functions from the `vibrations.py`. Then, the script then defines a function to solve the differential equations of motion using the `odeint` function from the `scipy.integrate` module, and functions to plot the system using the `matplotlib` module.






vibrations.ipynb: This Jupyter notebook contains interactive code and visualizations for the vibration systems.
