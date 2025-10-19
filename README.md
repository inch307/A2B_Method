# Adaptive Second-order Bounding Method

This repository contains the Python implementation for the paper Numerical Data Collection under Local Differential Privacy without Prior Knowledge.
The **Adaptive Second-order Bounding (A2B) method** is an online protocol for collecting numerical data under Local Differential Privacy (LDP) without prior knowledge of the data's true range. It adaptively tunes the data collection bounds $[l, r]$ to minimize the total estimation error by balancing the bias-variance trade-off.


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

## Hyperparameters

The behavior of the LB method can be controlled via the following command-line arguments. Each argument corresponds to a key parameter in our paper.

* `--alpha` (float, default: `0.05`)
    * The **target clipping ratio** $\alpha$. This is the desired proportion of data to be clipped on each side of the bounds. It serves as a direct control for the bias-variance trade-off. A smaller $\alpha$ aims for lower bias, while a larger $\alpha$ may reduce variance more aggressively.

* `--lr` (float, default: `0.3`)
    * The base **learning rate** $\eta$ for the bound update rule. It controls the magnitude of adjustments to $l_t$ and $r_t$ in each round.

* `--eps_ratio` (float, default: `0.7`)
    * The **privacy budget allocation ratio**. This value determines how the total privacy budget $\epsilon$ is split between the two signals sent by clients. `eps_ratio` of the budget is used for the LDP-protected numerical value, and `1 - eps_ratio` is used for the LDP-protected clipping status signal.

* `--steps` (int, default: `30`)
    * The total number of **rounds** $T$ for the iterative bounding process.

* `--zeta` (float, default: `0.3`)
    * The **numerical stability constant** $\zeta$. This small value is used in the denominator of the update rule to prevent division by zero when the estimated in-bound ratio $\hat{\theta}_{0,t}$ is close to zero.

* `--tau` (float, default: `2`)
    * The **error amplification parameter** $\tau$ ($\tau \ge 1$). It controls the non-linear response to the error term in the update rule. A value of $\tau > 1$ amplifies the update signal when the error is small, helping the system converge more robustly.
