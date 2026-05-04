const DEFAULT_EPS = 1e-8
const DEFAULT_MAXITER = 50
const DEFAULT_CHECK = true
const DEFAULT_CHECKEVERY = 5

struct ScalingConvergenceError <: Exception
    max_iter::Int
    eps::Float64
end

Base.showerror(io::IO, err::ScalingConvergenceError) =
    print(io, "scaling did not converge after ", err.max_iter, " iterations at eps=", err.eps)

@inline abs_power(x, ::Val{0}) = iszero(x) ? zero(abs(x)) : one(abs(x))
@inline abs_power(x, ::Val{1}) = abs(x)
@inline abs_power(x, ::Val{2}) = abs2(x)
@inline abs_power(x, ::Val{P}) where {P} = abs_power(x, P)
@inline abs_power(x, p) = abs(x)^p
@inline pow_arg(::Type, p::Val) = p
@inline pow_arg(::Type{T}, p::Real) where {T} = p == Inf ? Val(:inf) :
                                                p == 0 ? Val(0) :
                                                p == 1 ? Val(1) :
                                                p == 2 ? Val(2) : T(p)

inv_sqrt!(dst::AbstractVector{T}, src::AbstractVector{T}) where {T<:AbstractFloat} =
    dst .= ifelse.(iszero.(src), one(T), inv.(sqrt.(src)))

safe_inv!(dst::AbstractVector{T}, src::AbstractVector{T}) where {T<:AbstractFloat} =
    dst .= ifelse.(iszero.(src), one(T), inv.(src))

pow_inv!(dst::AbstractVector{<:AbstractFloat}, ::Val{1}) = dst
pow_inv!(dst::AbstractVector{T}, ::Val{P}) where {T<:AbstractFloat,P} =
    dst .= dst .^ inv(T(P))
pow_inv!(dst::AbstractVector{T}, p) where {T<:AbstractFloat} =
    dst .= dst .^ inv(T(p))

geometric_factors!(dst::AbstractVector{T}, mn::AbstractVector{T}, mx::AbstractVector{T}) where {T<:AbstractFloat} =
    dst .= ifelse.(iszero.(mx), one(T), inv.(sqrt.(mn) .* sqrt.(mx)))

norms_converged(nrms::AbstractVector{T}; eps = DEFAULT_EPS) where {T<:AbstractFloat} =
    mapreduce(n -> iszero(n) ? zero(T) : abs(one(T) - n), max, nrms; init = zero(T)) <= eps

struct SparseCOO{T,Ti<:Integer,V<:AbstractVector{T},Vi<:AbstractVector{Ti}} <: SparseArrays.AbstractSparseMatrix{T,Ti}
    m::Int
    n::Int
    rowval::Vi
    colval::Vi
    nzval::V
end

SparseCOO(m::Integer, n::Integer, rowval::Vi, colval::Vi, nzval::V) where {T,Ti,V<:AbstractVector{T},Vi<:AbstractVector{Ti}} =
    SparseCOO{T,Ti,V,Vi}(m, n, rowval, colval, nzval)

function SparseCOO(A::SparseArrays.SparseMatrixCSC)
    rowval, colval, nzval = SparseArrays.findnz(A)
    return SparseCOO(size(A, 1), size(A, 2), rowval, colval, nzval)
end

Base.size(A::SparseCOO) = (A.m, A.n)
SparseArrays.nnz(A::SparseCOO) = length(A.nzval)
SparseArrays.nonzeros(A::SparseCOO) = A.nzval
Base.copy(A::SparseCOO) = SparseCOO(A.m, A.n, copy(A.rowval), copy(A.colval), copy(A.nzval))

function Base.getindex(A::SparseCOO{T}, i::Integer, j::Integer) where {T}
    @inbounds for k in eachindex(A.nzval)
        A.rowval[k] == i && A.colval[k] == j && return A.nzval[k]
    end
    return zero(T)
end

storage_vector(A::AbstractMatrix{T}, n, value::T) where {T} = fill!(similar(A, T, n), value)
storage_vector(A::SparseCOO{T}, n, value::T) where {T} = fill!(similar(A.nzval, T, n), value)

scaling_vectors(A) = storage_vector(A, size(A, 1), one(eltype(A))), storage_vector(A, size(A, 2), one(eltype(A)))

@inline _nz_or_typemax(x, ::Type{T}) where {T} = iszero(x) ? typemax(T) : abs(x)

function row_col_maxabs!(rownrm::AbstractVector{T}, colnrm::AbstractVector{T}, A::AbstractMatrix{T}) where {T<:AbstractFloat}
    rownrm .= vec(mapreduce(abs, max, A; dims = 2, init = zero(T)))
    colnrm .= vec(mapreduce(abs, max, A; dims = 1, init = zero(T)))
    return rownrm, colnrm
end

function row_col_norm2!(rownrm::AbstractVector{T}, colnrm::AbstractVector{T}, A::AbstractMatrix{T}) where {T<:AbstractFloat}
    rownrm .= vec(mapreduce(identity, hypot, A; dims = 2, init = zero(T)))
    colnrm .= vec(mapreduce(identity, hypot, A; dims = 1, init = zero(T)))
    return rownrm, colnrm
end

function row_col_minmaxabs!(rowmin::AbstractVector{T}, rowmax::AbstractVector{T},
                            colmin::AbstractVector{T}, colmax::AbstractVector{T},
                            A::AbstractMatrix{T}) where {T<:AbstractFloat}
    rowmin .= vec(mapreduce(x -> _nz_or_typemax(x, T), min, A; dims = 2, init = typemax(T)))
    rowmax .= vec(mapreduce(abs, max, A; dims = 2, init = zero(T)))
    colmin .= vec(mapreduce(x -> _nz_or_typemax(x, T), min, A; dims = 1, init = typemax(T)))
    colmax .= vec(mapreduce(abs, max, A; dims = 1, init = zero(T)))
    rowmin .= ifelse.(iszero.(rowmax), zero(T), rowmin)
    colmin .= ifelse.(iszero.(colmax), zero(T), colmin)
    return rowmin, rowmax, colmin, colmax
end

function row_col_sum_abs_power!(rowsum::AbstractVector{T}, colsum::AbstractVector{T}, A::AbstractMatrix{T},
                                rowp, colp = rowp) where {T<:AbstractFloat}
    rowsum .= vec(mapreduce(x -> abs_power(x, rowp), +, A; dims = 2, init = zero(T)))
    colsum .= vec(mapreduce(x -> abs_power(x, colp), +, A; dims = 1, init = zero(T)))
    return rowsum, colsum
end

function row_col_maxabs!(rownrm::AbstractVector{T}, colnrm::AbstractVector{T}, A::SparseCOO{T}) where {T<:AbstractFloat}
    fill!(rownrm, zero(T)); fill!(colnrm, zero(T))
    @inbounds for k in eachindex(A.nzval)
        v = abs(A.nzval[k])
        i, j = A.rowval[k], A.colval[k]
        rownrm[i] = max(rownrm[i], v)
        colnrm[j] = max(colnrm[j], v)
    end
    return rownrm, colnrm
end

function row_col_norm2!(rownrm::AbstractVector{T}, colnrm::AbstractVector{T}, A::SparseCOO{T}) where {T<:AbstractFloat}
    fill!(rownrm, zero(T)); fill!(colnrm, zero(T))
    @inbounds for k in eachindex(A.nzval)
        v = A.nzval[k]
        i, j = A.rowval[k], A.colval[k]
        rownrm[i] = hypot(rownrm[i], v)
        colnrm[j] = hypot(colnrm[j], v)
    end
    return rownrm, colnrm
end

function row_col_minmaxabs!(rowmin::AbstractVector{T}, rowmax::AbstractVector{T},
                            colmin::AbstractVector{T}, colmax::AbstractVector{T},
                            A::SparseCOO{T}) where {T<:AbstractFloat}
    fill!(rowmin, typemax(T)); fill!(rowmax, zero(T))
    fill!(colmin, typemax(T)); fill!(colmax, zero(T))
    @inbounds for k in eachindex(A.nzval)
        a = abs(A.nzval[k])
        if !iszero(a)
            i, j = A.rowval[k], A.colval[k]
            rowmin[i] = min(rowmin[i], a)
            rowmax[i] = max(rowmax[i], a)
            colmin[j] = min(colmin[j], a)
            colmax[j] = max(colmax[j], a)
        end
    end
    @inbounds for i in eachindex(rowmin, rowmax)
        iszero(rowmax[i]) && (rowmin[i] = zero(T))
    end
    @inbounds for j in eachindex(colmin, colmax)
        iszero(colmax[j]) && (colmin[j] = zero(T))
    end
    return rowmin, rowmax, colmin, colmax
end

function row_col_sum_abs_power!(rowsum::AbstractVector{T}, colsum::AbstractVector{T}, A::SparseCOO{T},
                                rowp, colp = rowp) where {T<:AbstractFloat}
    fill!(rowsum, zero(T)); fill!(colsum, zero(T))
    @inbounds for k in eachindex(A.nzval)
        v = A.nzval[k]
        rowsum[A.rowval[k]] += abs_power(v, rowp)
        colsum[A.colval[k]] += abs_power(v, colp)
    end
    return rowsum, colsum
end

function row_norm2!(rownrm::AbstractVector{T}, A::AbstractMatrix{T}) where {T<:AbstractFloat}
    rownrm .= vec(mapreduce(identity, hypot, A; dims = 2, init = zero(T)))
    return rownrm
end
function col_norm2!(colnrm::AbstractVector{T}, A::AbstractMatrix{T}) where {T<:AbstractFloat}
    colnrm .= vec(mapreduce(identity, hypot, A; dims = 1, init = zero(T)))
    return colnrm
end
function row_sum_abs_power!(rowsum::AbstractVector{T}, A::AbstractMatrix{T}, p) where {T<:AbstractFloat}
    rowsum .= vec(mapreduce(x -> abs_power(x, p), +, A; dims = 2, init = zero(T)))
    return rowsum
end
function col_sum_abs_power!(colsum::AbstractVector{T}, A::AbstractMatrix{T}, p) where {T<:AbstractFloat}
    colsum .= vec(mapreduce(x -> abs_power(x, p), +, A; dims = 1, init = zero(T)))
    return colsum
end
function row_minmaxabs!(rowmin::AbstractVector{T}, rowmax::AbstractVector{T}, _, _, A::AbstractMatrix{T}) where {T<:AbstractFloat}
    rowmin .= vec(mapreduce(x -> _nz_or_typemax(x, T), min, A; dims = 2, init = typemax(T)))
    rowmax .= vec(mapreduce(abs, max, A; dims = 2, init = zero(T)))
    rowmin .= ifelse.(iszero.(rowmax), zero(T), rowmin)
    return rowmin, rowmax
end
function col_minmaxabs!(_, _, colmin::AbstractVector{T}, colmax::AbstractVector{T}, A::AbstractMatrix{T}) where {T<:AbstractFloat}
    colmin .= vec(mapreduce(x -> _nz_or_typemax(x, T), min, A; dims = 1, init = typemax(T)))
    colmax .= vec(mapreduce(abs, max, A; dims = 1, init = zero(T)))
    colmin .= ifelse.(iszero.(colmax), zero(T), colmin)
    return colmin, colmax
end

function row_norm2!(rownrm::AbstractVector{T}, A::SparseCOO{T}) where {T<:AbstractFloat}
    fill!(rownrm, zero(T))
    @inbounds for k in eachindex(A.nzval)
        rownrm[A.rowval[k]] = hypot(rownrm[A.rowval[k]], A.nzval[k])
    end
    return rownrm
end
function col_norm2!(colnrm::AbstractVector{T}, A::SparseCOO{T}) where {T<:AbstractFloat}
    fill!(colnrm, zero(T))
    @inbounds for k in eachindex(A.nzval)
        colnrm[A.colval[k]] = hypot(colnrm[A.colval[k]], A.nzval[k])
    end
    return colnrm
end
function row_sum_abs_power!(rowsum::AbstractVector{T}, A::SparseCOO{T}, p) where {T<:AbstractFloat}
    fill!(rowsum, zero(T))
    @inbounds for k in eachindex(A.nzval)
        rowsum[A.rowval[k]] += abs_power(A.nzval[k], p)
    end
    return rowsum
end
function col_sum_abs_power!(colsum::AbstractVector{T}, A::SparseCOO{T}, p) where {T<:AbstractFloat}
    fill!(colsum, zero(T))
    @inbounds for k in eachindex(A.nzval)
        colsum[A.colval[k]] += abs_power(A.nzval[k], p)
    end
    return colsum
end
function row_minmaxabs!(rowmin::AbstractVector{T}, rowmax::AbstractVector{T}, _, _, A::SparseCOO{T}) where {T<:AbstractFloat}
    fill!(rowmin, typemax(T)); fill!(rowmax, zero(T))
    @inbounds for k in eachindex(A.nzval)
        a = abs(A.nzval[k])
        if !iszero(a)
            i = A.rowval[k]
            rowmin[i] = min(rowmin[i], a)
            rowmax[i] = max(rowmax[i], a)
        end
    end
    @inbounds for i in eachindex(rowmin, rowmax)
        iszero(rowmax[i]) && (rowmin[i] = zero(T))
    end
    return rowmin, rowmax
end
function col_minmaxabs!(_, _, colmin::AbstractVector{T}, colmax::AbstractVector{T}, A::SparseCOO{T}) where {T<:AbstractFloat}
    fill!(colmin, typemax(T)); fill!(colmax, zero(T))
    @inbounds for k in eachindex(A.nzval)
        a = abs(A.nzval[k])
        if !iszero(a)
            j = A.colval[k]
            colmin[j] = min(colmin[j], a)
            colmax[j] = max(colmax[j], a)
        end
    end
    @inbounds for j in eachindex(colmin, colmax)
        iszero(colmax[j]) && (colmin[j] = zero(T))
    end
    return colmin, colmax
end

function scale_rows_cols!(A::AbstractMatrix, rscale::AbstractVector, cscale::AbstractVector)
    A .*= rscale .* transpose(cscale)
    return A
end
function scale_rows!(A::AbstractMatrix, rscale::AbstractVector)
    A .*= rscale
    return A
end
function scale_cols!(A::AbstractMatrix, cscale::AbstractVector)
    A .*= transpose(cscale)
    return A
end

function scale_rows_cols!(A::SparseCOO, rscale::AbstractVector, cscale::AbstractVector)
    A.nzval .*= view(rscale, A.rowval) .* view(cscale, A.colval)
    return A
end
function scale_rows!(A::SparseCOO, rscale::AbstractVector)
    A.nzval .*= view(rscale, A.rowval)
    return A
end
function scale_cols!(A::SparseCOO, cscale::AbstractVector)
    A.nzval .*= view(cscale, A.colval)
    return A
end

function run_iterative_scaling!(
    ws;
    max_iter::Integer = DEFAULT_MAXITER,
    check::Bool = DEFAULT_CHECK,
    check_every::Integer = DEFAULT_CHECKEVERY,
    eps::Real = DEFAULT_EPS,
    strict::Bool = false,
    step!,
    converged!,
    check_converged! = converged!,
)
    row, col = ws.scaling.row, ws.scaling.col
    fill!(row, one(eltype(row)))
    row !== col && fill!(col, one(eltype(col)))
    do_check = check && check_every > 0

    for k in 1:max_iter
        step!(ws)
        do_check && iszero(k % check_every) && check_converged!(ws; eps) && break
    end
    strict && !converged!(ws; eps) &&
        throw(ScalingConvergenceError(Int(max_iter), Float64(eps)))
    return ws.storage, ws.scaling
end
