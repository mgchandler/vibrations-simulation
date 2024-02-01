# vibration-system

This repository contains code for the demonstration of physical vibration systems under various conditions. It was developed for use with the Vibrations component of the Dynamics and Control unit at the University of Bristol.

The home directory contains two files of interest: the script `vibrations.py` which contains the simulation model and an interactive GUI, and the Jupyter notebook `Vibrations Simulation.ipynb` which walks through the simulation and runs the GUI.

The folder `individual_simulations` contains three non-interactive implementations of the vibrating system:
- `freevisc`: a free viscous model with equation $0 = cv + kx$.
- `freefric`: a free friction model with equation $0 = F_r v + kx$.
- `constforce`: a model which applies a constant force after some time has elapsed, with equation $F_0 = cv + kx$.

Each of these models has two implementations, identified with `partials` and `lambdas`. These indicate how the forces are implemented in Python: `partials` makes use of functions and fills individual arguments using `functools.partials`, whereas `lambdas` uses anonymous functions. These additional scripts are included as an example to the user of how you might go about simulating this more simply in Python, as the use of the `VibSimulation` class in the interactive version is quite complex, and complicated by the requirement to include widgets, plots, etc.

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

Download this repo (at the top of the page, click `<> Code` → `Download ZIP`) and extract the folder to your documents. Make sure that the files `vibrations.py` and `Vibrations Simulation.ipynb` are both located in the same directory. Open the Jupyter Notebook client, and open the file `Vibrations Simulation.ipynb` in Jupyter and run the code cell. This will produce an interactive simulation which allows the user to investigate the effects of each parameter on the vibrating system.
