# $\mbox{GPLVM}_+$

# What is this?
This is an implementation of the $\mbox{GPLVM}_+$ model presented in [Positive and Scale Invariant Gaussian Process Latent Variable Model for Astronomical Spectra, ESANN 2024](https://github.com/ngiann/GPLVMplus.jl/blob/main/ESANN2024.pdf).

# Installation

Apart from cloning, an easy way of using the package is the following:

1 - Add the registry [AINJuliaRegistry](https://github.com/HITS-AIN/AINJuliaRegistry)

2 - Switch into "package mode" with `]` and add the package with
```
add GPLVMplus
```

The following functions are of interest to the end user:
- `gplvmplus`, see [Experiment 1](#experiment-1).
- `inferlatent`, see [Inferring latent projections](#inferring-latent-projections).
- `predictivesampler`
  
# Demonstrations

Below we show two examples of how to use the model. The experiments demonstrate the use of the model with a dataset of 72 images that are made available via the package [GPLVMplusData.jl](https://github.com/HITS-AIN/GPLVMplusData.jl). These 72 images have been taken from the COIL-20 repository that can be found [here](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php). The 72 images are images of a rubber duck photographed from 72 angles, thus they form a latent space that is intrinsically a one-dimensional circle. We show in the experiments below that the model successfully discovers this latent space. 

## Experiment 1

In this experiment we run the model on the downsampled images setting the low-dimensional space to $Q=2$ dimensions.
We further specify the number of neurons for the neural network that parametrises the variational parameters for the posterior mean of the latent function values: we set the number of neurons in the first layer to 20 neurons with $H_1 = 20$. We also set the number of neurons in the second hidden layer also to 20 neurons with $H_2 = 20$. If $H_2$ is left unspecified, it is automatically set so that $H_2 = H_1$.

```
using GPLVMplus
using GPLVMplusData # must be independently installed
using PyPlot # must be independently installed.
             # Other plotting packages can be used instead

X = GPLVMplusData.loadducks(;every=4); # load rubber duck images in 32x32 resolution

# warmup
let
    gplvmplus(X; Q = 2, iterations = 1)
end

# Learn mapping from Q=2 latent dimensions to high-dimensional images.
# Use a two-hidden layer neural network for amortised inference. 
result = gplvmplus(X; Q = 2, H1 = 20, H2 = 20, iterations = 5000);

# Plot latent 2-dimensional projections
plot(result[:Z][1,:],result[:Z][2,:],"o")
```

## Experiment 2

This experiment demonstrates the scale-invariant property of the proposed $\mbox{GPLVM}_{+}$ model.
We reuse the same dataset of rubber ducks, taken from the [COIL-20 repository](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php).
This time, however, we first scale the images ny multiplying them with arbitrary positive coefficients.
The purpose of this experiment is to show that the model can discover the same latent space, as in [experiment 1](#experiment-1).

```
using GPLVMplus
using GPLVMplusData # must be independently installed
using PyPlot # must be independently installed.
             # Other plotting packages can be used instead
using Random

X = GPLVMplusData.loadducks(;every=4); # load rubber duck images in 32x32 resolution

# warmup
let
    gplvmplus(X; Q = 2, iterations = 1)
end

# Instantiate random number generator
rng = MersenneTwister(1);

# Sample 72 scaling coefficients between 0.5 and 2.5
C = rand(rng, 72)*2 .+ 0.5;

# Scale each image with the corresponding scaling coefficients
Xscale = reduce(hcat, [x*c for (x,c) in zip(eachcol(X),C)]);

# Learn mapping from Q=2 latent dimensions to high-dimensional scaled images.
result2 = gplvmplus(Xscale; Q = 2, H1 = 20, H2 = 20, iterations = 5000);

# Plot latent 2-dimensional projections
plot(result2[:Z][1,:],result2[:Z][2,:],"o")

# Compare inferred scaling coefficients to actual coefficients C
figure()
plot(C, label="scaling coefficients C")
plot(result2[:c], label="inferred scaling coefficients")
legend()
```

## Inferring latent projections

Continuing with the example above, we show how to infer the latent coordinate of a high-dimensional data item.
For convenience, we take one of the images used for training but scale it with a new scaling coefficient.

```
Xtest = 1.2345 * X[:,1]
```
We infer the latent coordinates, and associated scaling coefficient, using:
```
Ztest, ctest = inferlatent(Xtest, result2);
```
Barring local minima  present in the inference of the latent coordinate, variable `Ztest` should hold approximately the same latent coordinate as the training image `X[:,1]`:
```
display(Ztest)
display(result2[:Z][:,1])
```

## Sampling the predictive distribution
