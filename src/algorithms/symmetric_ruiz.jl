struct SymmetricRuizScaling{D}
    row::D
    col::D
end

struct SymmetricRuizWorkspace{S,D,W}
    scaling::S
    scale::D
    rownrm::D
    colnrm::D
    storage::W
end

symmetric_ruiz_equilibration(A; kwargs...) = symmetric_ruiz_equilibration!(copy(A); kwargs...)

function SymmetricRuizWorkspace(A)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("symmetric Ruiz scaling requires a square matrix"))
    d = storage_vector(A, size(A, 1), one(eltype(A)))
    return SymmetricRuizWorkspace(SymmetricRuizScaling(d, d), similar(d), similar(d), similar(d), A)
end

symmetric_ruiz_equilibration!(A; kwargs...) =
    symmetric_ruiz_equilibration!(A, SymmetricRuizWorkspace(A); kwargs...)

symmetric_ruiz_equilibration!(A, ws::SymmetricRuizWorkspace; kwargs...) =
    run_iterative_scaling!(ws; step! = symmetric_ruiz_step!,
        check_converged! = symmetric_ruiz_step_converged, converged! = symmetric_ruiz_converged, kwargs...)

function symmetric_ruiz_step!(ws::SymmetricRuizWorkspace)
    d = ws.scaling.row
    symmetric_norms!(ws.rownrm, ws.colnrm, ws.storage)
    inv_sqrt!(ws.scale, ws.rownrm)
    scale_rows_cols!(ws.storage, ws.scale, ws.scale)
    d .*= ws.scale
    return ws.storage
end

function symmetric_ruiz_step_converged(ws::SymmetricRuizWorkspace; eps = DEFAULT_EPS)
    return norms_converged(ws.scale; eps = eps / 2)
end

function symmetric_ruiz_converged(ws::SymmetricRuizWorkspace; eps = DEFAULT_EPS)
    symmetric_norms!(ws.rownrm, ws.colnrm, ws.storage)
    return norms_converged(ws.rownrm; eps)
end

function symmetric_norms!(dst, tmp, A)
    row_col_maxabs!(dst, tmp, A)
    dst .= max.(dst, tmp)
    return dst
end
