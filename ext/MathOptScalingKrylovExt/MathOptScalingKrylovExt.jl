module MathOptScalingKrylovExt

using Krylov: cgls, crls, lslq, lsmr, lsqr
using SparseArrays: SparseArrays, sparse
using MathOptScaling
import MathOptScaling: SparseCOO, CurtisReidWorkspace, scale_rows_cols!

const MOS = MathOptScaling

_solve(s::Symbol, F, b; kwargs...) = _solve(Val(s), F, b; kwargs...)
_solve(::Val{:lsmr}, F, b; kwargs...) = lsmr(F, b; kwargs...)
_solve(::Val{:lsqr}, F, b; kwargs...) = lsqr(F, b; kwargs...)
_solve(::Val{:lslq}, F, b; kwargs...) = lslq(F, b; kwargs...)
_solve(::Val{:cgls}, F, b; kwargs...) = cgls(F, b; kwargs...)
_solve(::Val{:crls}, F, b; kwargs...) = crls(F, b; kwargs...)
_solve(::Val{S}, F, b; kwargs...) where {S} =
    throw(ArgumentError("unsupported Krylov solver $(repr(S)); use one of :lsmr, :lsqr, :lslq, :cgls, :crls"))

function _ls_problem(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    m, n = size(A)
    rows, cols, vals, rhs = Int[], Int[], T[], T[]
    @inbounds for j in 1:n, i in 1:m
        v = A[i, j]
        iszero(v) && continue
        k = length(rhs) + 1
        push!(rows, k); push!(cols, i); push!(vals, one(T))
        push!(rows, k); push!(cols, m + j); push!(vals, one(T))
        push!(rhs, -log(abs(v)))
    end
    return sparse(rows, cols, vals, length(rhs), m + n), rhs
end

function _ls_problem(A::SparseCOO{T}) where {T<:AbstractFloat}
    rows, cols, vals, rhs = Int[], Int[], T[], T[]
    @inbounds for k in eachindex(A.nzval)
        v = A.nzval[k]
        iszero(v) && continue
        kk = length(rhs) + 1
        push!(rows, kk); push!(cols, A.rowval[k]); push!(vals, one(T))
        push!(rows, kk); push!(cols, A.m + A.colval[k]); push!(vals, one(T))
        push!(rhs, -log(abs(v)))
    end
    return sparse(rows, cols, vals, length(rhs), A.m + A.n), rhs
end

function MOS.curtis_reid_scaling!(A, ws::CurtisReidWorkspace; solver = Val(:lsmr), kwargs...)
    drow, dcol = ws.scaling.row, ws.scaling.col
    F, b = _ls_problem(ws.storage)
    x, _ = _solve(solver, F, b; kwargs...)
    m = size(ws.storage, 1)
    rscale = exp.(view(x, 1:m))
    cscale = exp.(view(x, m+1:length(x)))
    scale_rows_cols!(ws.storage, rscale, cscale)
    drow .*= rscale
    dcol .*= cscale
    return ws.storage, ws.scaling
end

end
