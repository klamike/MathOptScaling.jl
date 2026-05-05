include(joinpath(@__DIR__, "..", "..", "shared.jl"))

using Metal

if Metal.functional()
    run_backend_tests("Metal", MtlArray, Float32; curtis_reid = false)
else
    @info "Metal is not functional; skipping Metal tests"
end
