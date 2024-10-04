function unpack_gplvmplus(p, D, N, net, Q)

    nwts = numweights(net)
    
    @assert(length(p) == Q*N + 1 + 1 + nwts + N + 2 + N)

    MARK = 0

    Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N  # latent coordinates

    θ = log1pexp(p[MARK+1]); MARK += 1                  # GP lengthscale

    β = log1pexp(p[MARK+1]) + 1; MARK += 1              # inverse noise

    w = p[MARK+1:MARK+nwts]; MARK += nwts               # neural network weights

    Λroot = Diagonal((p[MARK+1:MARK+N])); MARK += N     # diagonal for parametrising covariance of posterior function values

    α = log1pexp(p[MARK+1]); MARK += 1                  # scaling coefficient inside exp(⋅) non-linearity

    b = p[MARK+1]; MARK += 1                            # shift coefficient inside exp(⋅) non-linearity

    c = log1pexp.(p[MARK+1:MARK+N]); MARK += N          # individual scaling coefficients

    @assert(MARK == length(p))

    μ = net(w, Z)                                       # posterior mean of latent function values parametrised by neural network
    
    return Z, [1.0;θ], β, μ, Λroot, w, α, b, c          # global amplitude fixed to 1.0 without loss of generalisation

end