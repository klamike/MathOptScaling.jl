struct CurtisReidScaling{R,C}
    row::R
    col::C
end

struct CurtisReidWorkspace{S,R,C,W}
    scaling::S
    rscale::R
    cscale::C
    storage::W
end

function CurtisReidWorkspace(A)
    drow, dcol = scaling_vectors(A)
    return CurtisReidWorkspace(CurtisReidScaling(drow, dcol), similar(drow), similar(dcol), A)
end

function _build_ls_matrix end

curtis_reid_scaling(A; kwargs...) = curtis_reid_scaling!(copy(A); kwargs...)
curtis_reid_scaling!(A; kwargs...) = curtis_reid_scaling!(A, CurtisReidWorkspace(A); kwargs...)
curtis_reid_scaling!(A, ws; kwargs...) = error("Curtis Reid requires Krylov.jl (add `using Krylov`)")
