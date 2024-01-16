# vibration-system

This repository contains code for the demonstration of physical vibration systems under various conditions. It was developed for use with the Vibrations component of the Dynamics and Control unit at the University of Bristol.

The home directory contains two files of interest: the script `vibrations.py` which contains the simulation model and an interactive GUI, and the Jupyter notebook `Vibrations Simulation.ipynb` which walks through the simulation and instatiates the GUI.

## Using this simulation

This project is implemented in Python, and the following modules are required in the user's environment:
- `numpy`
- `scipy`
- `matplotlib`
- `jupyter`
- `ipywidgets`
- `IPython`
- `sympy`
All of these modules are available in the default installation which comes with Anaconda.

Download this repo and load the Jupyter client. Make sure that the files `vibrations.py` and `Vibrations Simulation.ipynb` are both found in the same directory. Open the file `Vibrations Simulation.ipynb` and run the code cell. This will produce an interactive simulation which allows the user to investigate the effects of each parameter on the vibrating system.