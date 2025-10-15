# Lyapunov Bounding Method

This repository contains the Python implementation for the paper Numerical Data Collection under Local Differential Privacy without Prior Knowledge.
The **Lyapunov Bounding (LB) method** is an online protocol for collecting numerical data under Local Differential Privacy (LDP) without prior knowledge of the data's true range. It adaptively tunes the data collection bounds $[l, r]$ to minimize the total estimation error by balancing the bias-variance trade-off.


## Reproducing Experiments
To reproduce the main experimental results from the paper, you can run the provided script. 
```
$ ./exp.ps1
```

## Running a Single Simulation
To run a single instance of the simulation with custom hyperparameters, execute the EXP.py.
```
$ python EXP.py --dataset adult --mech pmsub --alpha 0.05 --lr 0.3 --steps 30 --tau 2 --zeta 0.1 --eps_ratio 0.7
```

