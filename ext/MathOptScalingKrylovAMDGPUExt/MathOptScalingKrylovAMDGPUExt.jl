module MathOptScalingKrylovAMDGPUExt

using AMDGPU: ROCVector, AMDGPU
using AMDGPU.rocSPARSE: ROCSparseMatrixCOO
using MathOptScaling
const MOS = MathOptScaling

function MOS._build_ls_matrix(::Type{T}, rowval::ROCVector{Ti}, colval::ROCVector{Ti}, m::Integer, n::Integer) where {T<:AbstractFloat,Ti}
    nz = length(rowval)
    rows = ROCVector{Ti}(undef, 2nz)
    cols = ROCVector{Ti}(undef, 2nz)
    rows[1:2:end] .= 1:nz
    rows[2:2:end] .= 1:nz
    cols[1:2:end] .= rowval
    cols[2:2:end] .= colval .+ Ti(m)
    return ROCSparseMatrixCOO{T,Ti}(rows, cols, AMDGPU.ones(T, 2nz), (Int(nz), Int(m + n)))
end

end
