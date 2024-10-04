module GPLVMplus

    using LinearAlgebra, Distributions, Random, Statistics, Distances, LogExpFunctions
    using ExponentialExpectations, FillArrays
    # using ForwardNeuralNetworks
    using Printf, PyPlot
    using JLD2
    # using Artifacts, LazyArtifacts
    using Transducers
    using Optimization, OptimizationOptimJL, OptimizationBBO, Zygote#, LineSearches

    # simple neural net
    include("neuralnets/TwoLayerNetwork.jl")
    include("neuralnets/ThreeLayerNetwork.jl")

    # common
    include("common/covariance.jl" )
    include("common/woodbury.jl")
    # include("common/rbf.jl")
    include("common/myskip.jl")
    include("common/inverterrors.jl")
    include("common/toydata.jl")
    include("common/expectation_latent_function_values.jl")
    include("common/expectation_of_sum_D_log_prior_zero_mean.jl")
    include("common/entropy.jl")
    include("common/getfg!.jl")
    include("common/callback.jl")

    # GPLVM₊
    include("gplvmplus/gplvmplus.jl")
    include("gplvmplus/predictivesampler.jl")
    include("gplvmplus/marginallikelihood.jl")
    include("gplvmplus/marginallikelihood_VERIFY.jl")
    include("gplvmplus/infertestlatent.jl")
    include("gplvmplus/inferlightcurve.jl")
    include("gplvmplus/infertestlatent_photo.jl")
    include("gplvmplus/numerically_VERIFY.jl")
    include("gplvmplus/unpack_gplvmplus.jl")
    include("gplvmplus/unpack_inferlatent_gplvmplus.jl")
    include("gplvmplus/partial_objective.jl")

    export gplvmplus
    
end
