# Data

Data used in the repeatability code is stored here. We utilize data generated
from several sources in the repeatability package. An explanation of the data
sets are provided below.

## Linearized Cart-Pole

The linearized cart-pole data is generated using a system description in
[NNV](https://github.com/verivital/nnv) with a neural network controller. The
`CartPoleSamples_Linearized.mat` file consists of 12234 initial states X and the
resulting states Y generated from the system. For code to generate more samples,
see the [NNV](https://github.com/verivital/nnv) repository.

## Nonlinear Cart-Pole

The nonlinear cart-pole data is generated using a system description in
[NNV](https://github.com/verivital/nnv) with a neural network controller. The
`CartPoleSamples_Nonlinear.mat` file consists of 20020 initial states X and the
resulting states Y generated from the system. For code to generate more samples,
see the [NNV](https://github.com/verivital/nnv) repository.

## Dynamic Programming (FHT & THT)

We also include data pre-generted from SReachTools for speed. The
`DynamicProgrammingFHT.mat` and `DynamicProgrammingTHT.mat` files contain the
results of the dynamic programming algorithm in SReachTools for a double
integrator system over a time horizon of N = 5 for the first-hitting time
problem (FHT) and the terminal-hitting time problem (THT). The results are
stored in a 5x100x100 matrix called `PrDP`, which contains the safety
probabilities at each time index, where the first index of `PrDP` corresponds to
the time index.
