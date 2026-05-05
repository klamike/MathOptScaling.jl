struct CurtisReidScaling{R,C}
    row::R
    col::C
end

struct CurtisReidWorkspace{S,W}
    scaling::S
    storage::W
end

function CurtisReidWorkspace(A)
    drow, dcol = scaling_vectors(A)
    return CurtisReidWorkspace(CurtisReidScaling(drow, dcol), A)
end

curtis_reid_scaling(A; kwargs...) = curtis_reid_scaling!(copy(A); kwargs...)
curtis_reid_scaling!(A; kwargs...) = curtis_reid_scaling!(A, CurtisReidWorkspace(A); kwargs...)
curtis_reid_scaling!(A, ws; kwargs...) = error("Curtis Reid requires Krylov.jl (add `using Krylov`)")
