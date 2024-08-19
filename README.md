# $\mbox{GPLVM}_+$

Implementation of the $\mbox{GPLVM}_+$ model presented in *Positive and Scale Invariant Gaussian Process Latent Variable Model for Astronomical Spectra, ESANN 2024*.

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