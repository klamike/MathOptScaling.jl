include(joinpath(@__DIR__, "..", "..", "shared.jl"))

using CUDA
using Krylov  # loads MathOptScalingKrylovExt + MathOptScalingKrylovCUDAExt

if CUDA.functional()
    run_backend_tests("CUDA", CuArray, Float32)
    run_backend_tests("CUDA", CuArray, Float64)
else
    @info "CUDA is not functional; skipping CUDA tests"
end
