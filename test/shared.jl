using LinearAlgebra: Diagonal, I, Symmetric, diag
using SparseArrays: SparseArrays, sparse, spzeros, SparseMatrixCSC
using Test
using MathOptScaling:
    SparseCOO, ScalingConvergenceError,
    RuizScaling, RuizWorkspace, ruiz_equilibration, ruiz_equilibration!, ruiz_converged,
    SymmetricRuizScaling, SymmetricRuizWorkspace, symmetric_ruiz_equilibration, symmetric_ruiz_equilibration!,
    DiagonalScaling, DiagonalNormWorkspace, diagonal_norm_scaling, diagonal_norm_scaling!,
    ChambollePockScaling, ChambollePockWorkspace, chambolle_pock_scaling, chambolle_pock_scaling!,
    TomlinScaling, TomlinWorkspace, tomlin_scaling, tomlin_scaling!, tomlin_converged,
    CurtisReidScaling, CurtisReidWorkspace, curtis_reid_scaling, curtis_reid_scaling!,
    CurtisReidScaling, CurtisReidWorkspace, curtis_reid_scaling, curtis_reid_scaling!

include("matrices.jl")

dense_arr(to, M) = to(Matrix(M))
function coo_arr(to, M)
    S = M isa SparseMatrixCSC ? M : sparse(M)
    r, c, v = SparseArrays.findnz(S)
    return SparseCOO(size(S, 1), size(S, 2), to(r), to(c), to(v))
end

function densify(A)
    A isa SparseCOO || return Array(A)
    M = zeros(eltype(A), size(A))
    r, c, v = Array(A.rowval), Array(A.colval), Array(A.nzval)
    @inbounds for k in eachindex(v)
        M[r[k], c[k]] = v[k]
    end
    return M
end

reconstructs(S, A, D) =
    densify(S) ≈ Diagonal(Array(D.row)) * Matrix(A) * Diagonal(Array(D.col))

function run_backend_tests(label, to, T = Float64; eps = sqrt(Base.eps(T)), curtis_reid = true)
    forms = (("dense", M -> dense_arr(to, M)), ("COO", M -> coo_arr(to, M)))
    algos = (
        ("Ruiz",            X -> ruiz_equilibration!(X; eps)),
        ("DiagonalNorm",    X -> diagonal_norm_scaling!(X; p = 2)),
        ("ChambollePock-0", X -> chambolle_pock_scaling!(X; alpha = 0)),
        ("ChambollePock-1", X -> chambolle_pock_scaling!(X; alpha = 1)),
        ("ChambollePock-2", X -> chambolle_pock_scaling!(X; alpha = 2)),
        ("Tomlin",          X -> tomlin_scaling!(X; check = false)),
    )
    coo_only_algos = curtis_reid ? (
        ("CurtisReid:lsmr", X -> curtis_reid_scaling!(X; solver = :lsmr)),
        ("CurtisReid:lsqr", X -> curtis_reid_scaling!(X; solver = Val(:lsqr))),
        ("CurtisReid:lslq", X -> curtis_reid_scaling!(X; solver = :lslq)),
        ("CurtisReid:cgls", X -> curtis_reid_scaling!(X; solver = :cgls)),
        ("CurtisReid:crls", X -> curtis_reid_scaling!(X; solver = Val(:crls))),
    ) : ()
    @testset "$label $T" begin
        for (i, M) in enumerate(MATRICES)
            A = T.(M)
            @testset "M$i $fname / $aname" for (fname, mk) in forms, (aname, run!) in algos
                S, D = run!(mk(A))
                @test reconstructs(S, A, D)
            end
            @testset "M$i COO / $aname" for (aname, run!) in coo_only_algos
                S, D = run!(coo_arr(to, A))
                @test reconstructs(S, A, D)
            end
            size(M, 1) == size(M, 2) || continue
            H = T.(Matrix(Symmetric(M)))
            @testset "M$i SymmetricRuiz / $fname" for (fname, mk) in forms
                S, D = symmetric_ruiz_equilibration!(mk(H); eps)
                @test densify(S) ≈ Diagonal(Array(D.row)) * H * Diagonal(Array(D.row))
            end
        end
    end
end
