include(joinpath(@__DIR__, "..", "..", "shared.jl"))

using JLArrays: jl

run_backend_tests("JLArray", jl, Float32; curtis_reid = false)
run_backend_tests("JLArray", jl, Float64; curtis_reid = false)
