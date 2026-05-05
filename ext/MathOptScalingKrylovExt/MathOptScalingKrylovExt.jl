module MathOptScalingKrylovExt

using Krylov: cgls, crls, lslq, lsmr, lsqr
using SparseArrays: sparse
using MathOptScaling
import MathOptScaling: SparseCOO, CurtisReidWorkspace, scale_rows_cols!, _build_ls_matrix

const MOS = MathOptScaling

_solve(s::Symbol, F, b; kwargs...) = _solve(Val(s), F, b; kwargs...)
_solve(::Val{:lsmr}, F, b; kwargs...) = lsmr(F, b; kwargs...)
_solve(::Val{:lsqr}, F, b; kwargs...) = lsqr(F, b; kwargs...)
_solve(::Val{:lslq}, F, b; kwargs...) = lslq(F, b; kwargs...)
_solve(::Val{:cgls}, F, b; kwargs...) = cgls(F, b; kwargs...)
_solve(::Val{:crls}, F, b; kwargs...) = crls(F, b; kwargs...)
_solve(::Val{S}, F, b; kwargs...) where {S} =
    throw(ArgumentError("unsupported Krylov solver $(repr(S)); use one of :lsmr, :lsqr, :lslq, :cgls, :crls"))

function MOS._build_ls_matrix(::Type{T}, rowval::Vector{Ti}, colval::Vector{Ti}, m::Integer, n::Integer) where {T<:AbstractFloat,Ti}
    nz = length(rowval)
    rows = Vector{Ti}(undef, 2nz)
    cols = Vector{Ti}(undef, 2nz)
    vals = ones(T, 2nz)
    @inbounds for k in 1:nz
        rows[2k - 1] = k; cols[2k - 1] = rowval[k]
        rows[2k]     = k; cols[2k]     = m + colval[k]
    end
    return sparse(rows, cols, vals, nz, m + n)
end

MOS._build_ls_matrix(::Type{T}, rowval::AbstractVector, colval::AbstractVector, m, n) where {T} =
    throw(ArgumentError("Curtis Reid only supports CPU, CUDA, and AMDGPU."))

function MOS.curtis_reid_scaling!(A, ws::CurtisReidWorkspace; solver = Val(:lsmr), kwargs...)
    ws.storage isa SparseCOO || throw(ArgumentError("curtis_reid_scaling! requires a SparseCOO input"))
    drow, dcol = ws.scaling.row, ws.scaling.col
    storage = ws.storage
    T = eltype(storage.nzval)
    F = MOS._build_ls_matrix(T, storage.rowval, storage.colval, storage.m, storage.n)
    b = .-log.(abs.(storage.nzval))
    x, _ = _solve(solver, F, b; kwargs...)
    m = size(storage, 1)
    copyto!(ws.rscale, exp.(view(x, 1:m)))
    copyto!(ws.cscale, exp.(view(x, m+1:length(x))))
    scale_rows_cols!(storage, ws.rscale, ws.cscale)
    drow .*= ws.rscale
    dcol .*= ws.cscale
    return storage, ws.scaling
end

end
