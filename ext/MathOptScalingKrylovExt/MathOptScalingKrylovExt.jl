module MathOptScalingKrylovExt

using Krylov: lsmr
using SparseArrays: SparseArrays, sparse
using MathOptScaling
import MathOptScaling: SparseCOO, CurtisReidWorkspace, scale_rows_cols!

const MOS = MathOptScaling

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

function MOS.curtis_reid_scaling!(A, ws::CurtisReidWorkspace; kwargs...)
    drow, dcol = ws.scaling.row, ws.scaling.col
    F, b = _ls_problem(ws.storage)
    x, _ = lsmr(F, b; kwargs...)
    m = size(ws.storage, 1)
    rscale = exp.(view(x, 1:m))
    cscale = exp.(view(x, m+1:length(x)))
    scale_rows_cols!(ws.storage, rscale, cscale)
    drow .*= rscale
    dcol .*= cscale
    return ws.storage, ws.scaling
end

end
