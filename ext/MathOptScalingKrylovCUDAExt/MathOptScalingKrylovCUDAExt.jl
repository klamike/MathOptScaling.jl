module MathOptScalingKrylovCUDAExt

using CUDA: CuVector, CUDA
using CUDA.CUSPARSE: CuSparseMatrixCOO
using MathOptScaling
const MOS = MathOptScaling

function MOS._build_ls_matrix(::Type{T}, rowval::CuVector{Ti}, colval::CuVector{Ti}, m::Integer, n::Integer) where {T<:AbstractFloat,Ti}
    nz = length(rowval)
    rows = CuVector{Ti}(undef, 2nz)
    cols = CuVector{Ti}(undef, 2nz)
    rows[1:2:end] .= 1:nz
    rows[2:2:end] .= 1:nz
    cols[1:2:end] .= rowval
    cols[2:2:end] .= colval .+ Ti(m)
    return CuSparseMatrixCOO{T,Ti}(rows, cols, CUDA.ones(T, 2nz), (Int(nz), Int(m + n)))
end

end
