# $\mbox{GPLVM}_+$

Implementation of the $\mbox{GPLVM}_+$ model presented in *Positive and Scale Invariant Gaussian Process Latent Variable Model for Astronomical Spectra, ESANN 2024*.

Below we show two examples of how to use the model. The experiments demonstrate the use of the model with a dataset of 72 images that are made available via the package [GPLVMplusData.jl](https://github.com/HITS-AIN/GPLVMplusData.jl). The 72 images are images of a rubber duck photographed from 72 angles, thus they form a latent space that is intrinsically a one-dimensional circle. We show in the experiments below that the model successfully discovers this latent space. 

## Experiment 1

In this experiment we run the model on the downsampled images setting the low-dimensional space to $Q=2$ dimensions.
We further specify the number of neurons for the neural network that parametrises the variational parameters for the posterior mean of the latent function values: we set the number of neurons in the first layer to 20 neurons with $H_1 = 20$; we also set the number of neurons in the second hidden layer also to 20 neurons with $H_2 = 20$. If $H_2$ is left unspecified, it is automatically set so that $H_2 = H_1$.

```
using GPLVMplus
using GPLVMplusData # must be independently installed

X = GPLVMplusData.loadducks(;every=4); # load rubber duck images in 32x32 resolution

# warmup
let
    gplvmplus(X; Q = 2, iterations = 1)
end

# Learn mapping from Q=2 latent dimensions to high-dimensional images.
# Use a two-hidden layer neural network for amortised inference. 
result = gplvmplus(X; Q = 2, H1 = 20, H2 = 20, iterations = 5000);

# Plot latent coordinates
using PyPlot # must be independently installed
plot(result[:Z][1,:],result[:Z][2,:],"o")
```

## Experiment 2
