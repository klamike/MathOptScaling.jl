# MathOptScaling

Scaling routines for sparse and dense matrices on CPU and GPU.

```julia
using LinearAlgebra: Diagonal
using MathOptScaling: ruiz_equilibration

A = [
    100.0  0.0   3.0;
     2.0   4.0   0.0;
    -1.5   5.0   0.7;
]
S, D = ruiz_equilibration(A)
@assert S ≈ Diagonal(D.row) * A * Diagonal(D.col)
@assert all(≈(1), maximum(abs, S; dims=1)) && all(≈(1), maximum(abs, S; dims=2))
```
