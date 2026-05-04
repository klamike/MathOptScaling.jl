module MathOptScalingKernelAbstractionsExt

using Atomix
using GPUArrays
using KernelAbstractions
using MathOptScaling
import MathOptScaling: abs_power, SparseCOO

const MOS = MathOptScaling

const GPUSparseCOO{T} = SparseCOO{T,Ti,V,Vi} where {Ti,V<:GPUArrays.AbstractGPUVector{T},Vi<:GPUArrays.AbstractGPUVector{Ti}}

include("kernels.jl")

function coo_atomic_reduce!(rowdst::V, coldst::V, A::GPUSparseCOO{T}, rowp, colp, op) where {T,V<:GPUArrays.AbstractGPUVector{T}}
    fill!(rowdst, zero(T)); fill!(coldst, zero(T))
    coo_atomic_reduce_kernel!(get_backend(A.nzval))(
        rowdst, coldst, A.rowval, A.colval, A.nzval, rowp, colp, op;
        ndrange = length(A.nzval),
    )
    return rowdst, coldst
end

MOS.row_col_maxabs!(row::GPUArrays.AbstractGPUVector{T}, col::GPUArrays.AbstractGPUVector{T}, A::GPUSparseCOO{T}) where {T<:AbstractFloat} =
    coo_atomic_reduce!(row, col, A, Val(1), Val(1), Val(:max))
MOS.row_col_norm2!(row::GPUArrays.AbstractGPUVector{T}, col::GPUArrays.AbstractGPUVector{T}, A::GPUSparseCOO{T}) where {T<:AbstractFloat} =
    coo_atomic_reduce!(row, col, A, Val(1), Val(1), Val(:hypot))
MOS.row_col_sum_abs_power!(row::GPUArrays.AbstractGPUVector{T}, col::GPUArrays.AbstractGPUVector{T}, A::GPUSparseCOO{T}, rowp, colp = rowp) where {T<:AbstractFloat} =
    coo_atomic_reduce!(row, col, A, rowp, colp, Val(:sum))

function MOS.row_col_minmaxabs!(rowmin::GPUArrays.AbstractGPUVector{T}, rowmax::GPUArrays.AbstractGPUVector{T},
                                 colmin::GPUArrays.AbstractGPUVector{T}, colmax::GPUArrays.AbstractGPUVector{T},
                                 A::GPUSparseCOO{T}) where {T<:AbstractFloat}
    fill!(rowmin, typemax(T)); fill!(rowmax, zero(T))
    fill!(colmin, typemax(T)); fill!(colmax, zero(T))
    coo_minmax_kernel!(get_backend(A.nzval))(
        rowmin, rowmax, colmin, colmax, A.rowval, A.colval, A.nzval; ndrange = length(A.nzval)
    )
    rowmin .= ifelse.(iszero.(rowmax), zero(T), rowmin)
    colmin .= ifelse.(iszero.(colmax), zero(T), colmin)
    return rowmin, rowmax, colmin, colmax
end

function coo_axis_reduce!(dst::V, idx, A::GPUSparseCOO{T}, p, op) where {T,V<:GPUArrays.AbstractGPUVector{T}}
    fill!(dst, zero(T))
    coo_axis_reduce_kernel!(get_backend(A.nzval))(dst, idx, A.nzval, p, op; ndrange = length(A.nzval))
    return dst
end

MOS.row_norm2!(row::GPUArrays.AbstractGPUVector{T}, A::GPUSparseCOO{T}) where {T<:AbstractFloat} =
    coo_axis_reduce!(row, A.rowval, A, Val(1), Val(:hypot))
MOS.col_norm2!(col::GPUArrays.AbstractGPUVector{T}, A::GPUSparseCOO{T}) where {T<:AbstractFloat} =
    coo_axis_reduce!(col, A.colval, A, Val(1), Val(:hypot))
MOS.row_sum_abs_power!(row::GPUArrays.AbstractGPUVector{T}, A::GPUSparseCOO{T}, p) where {T<:AbstractFloat} =
    coo_axis_reduce!(row, A.rowval, A, p, Val(:sum))
MOS.col_sum_abs_power!(col::GPUArrays.AbstractGPUVector{T}, A::GPUSparseCOO{T}, p) where {T<:AbstractFloat} =
    coo_axis_reduce!(col, A.colval, A, p, Val(:sum))

function _coo_axis_minmax!(dstmin::V, dstmax::V, idx, A::GPUSparseCOO{T}) where {T,V<:GPUArrays.AbstractGPUVector{T}}
    fill!(dstmin, typemax(T)); fill!(dstmax, zero(T))
    coo_axis_minmax_kernel!(get_backend(A.nzval))(dstmin, dstmax, idx, A.nzval; ndrange = length(A.nzval))
    dstmin .= ifelse.(iszero.(dstmax), zero(T), dstmin)
    return dstmin, dstmax
end
MOS.row_minmaxabs!(rowmin::GPUArrays.AbstractGPUVector{T}, rowmax::GPUArrays.AbstractGPUVector{T}, _, _, A::GPUSparseCOO{T}) where {T<:AbstractFloat} =
    _coo_axis_minmax!(rowmin, rowmax, A.rowval, A)
MOS.col_minmaxabs!(_, _, colmin::GPUArrays.AbstractGPUVector{T}, colmax::GPUArrays.AbstractGPUVector{T}, A::GPUSparseCOO{T}) where {T<:AbstractFloat} =
    _coo_axis_minmax!(colmin, colmax, A.colval, A)

end
