The 2D Ising model stands as a momentous model in physics, showcasing a genuine phase transition. It
was solved exactly by Ernst Ising (1D) in 1920 and in 2D analytically solved by Onsager(2D)4 in 1944. We
here present the methods to solve Ising model using three different approaches. Due to its lack of intrinsic
dynamical evolution, its simulation relies on Monte Carlo algorithms. Particularly near its phase transi-
tion, the Metropolis algorithm’s efficiency declines sharply, a limitation circumvented by the SW and Wolff
method. Critical exponents describe this phase transition, their determination and comparison to theo-
retical values form the bases of this report. Finally, the performance of all three algorithms is compared.
We then apply the use of these cluster algorithms to Hard sphere model, and presented its comparison
with theoritical description, concluding this study which is conducted as part of the Computational Physics
course (PHY612N) taught by Dr. Sunil Pratap Singh at the Department of Physics, IISER Bhopal.

Programming language : Python

## Dependencies
- Python 3.x
- NumPy
- Matplotlib
- Numba

## Overview

The 2D Ising model consists of a lattice of spins that can take values of +1 or -1. Each spin interacts with its nearest neighbors, and the system evolves according to the Metropolis algorithm. The simulation computes various physical quantities such as energy, magnetization, and specific heat as a function of temperature.

The Hamiltonian \( H \) for the 2D Ising model in the presence of an external magnetic field \( B \) is given by the following expression:

\[ H = -J \sum_{\langle i, j \rangle} s_i s_j - \mu B \sum_i s_i \]

Here, \( s_i \) represents the spin at site \( i \), which can take values of +1 or -1. \( J \) is the interaction strength between nearest neighbor spins \( \langle i, j \rangle \), \( \mu \) is the magnetic moment of each spin, and \( B \) is the strength of the external magnetic field. The first sum runs over all nearest neighbor pairs of spins, while the second sum runs over all spins in the lattice.

$Features$

- Implements the Metropolis algorithm for efficient Monte Carlo sampling
- Compares the sampling with both the Wolff and SW Algorithm
- Utilizes Numba JIT compilation for accelerated performance
- Computes energy, magnetization, and specific heat of the system
- Allows customization of lattice size, number of Monte Carlo sweeps, temperature range, and external magnetic field strength
- Provides visualizations of the computed physical quantities






