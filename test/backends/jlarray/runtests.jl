include(joinpath(@__DIR__, "..", "..", "shared.jl"))

using JLArrays: jl

run_backend_tests("JLArray", jl, Float32)
run_backend_tests("JLArray", jl, Float64)
