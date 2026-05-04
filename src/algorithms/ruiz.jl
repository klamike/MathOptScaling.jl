struct RuizScaling{R,C}
    row::R
    col::C
end

struct RuizWorkspace{S,R,C,W}
    scaling::S
    rscale::R
    cscale::C
    rownrm::R
    colnrm::C
    storage::W
end

ruiz_equilibration(A; kwargs...) = ruiz_equilibration!(copy(A); kwargs...)

function RuizWorkspace(A)
    drow, dcol = scaling_vectors(A)
    return RuizWorkspace(
        RuizScaling(drow, dcol),
        similar(drow), similar(dcol),
        similar(drow), similar(dcol),
        A,
    )
end

ruiz_equilibration!(A; kwargs...) = ruiz_equilibration!(A, RuizWorkspace(A); kwargs...)

ruiz_equilibration!(A, ws::RuizWorkspace; kwargs...) =
    run_iterative_scaling!(ws; step! = ruiz_step!,
        check_converged! = ruiz_step_converged, converged! = ruiz_converged, kwargs...)

function ruiz_step!(ws::RuizWorkspace)
    drow, dcol = ws.scaling.row, ws.scaling.col
    row_col_maxabs!(ws.rownrm, ws.colnrm, ws.storage)
    inv_sqrt!(ws.rscale, ws.rownrm)
    inv_sqrt!(ws.cscale, ws.colnrm)
    scale_rows_cols!(ws.storage, ws.rscale, ws.cscale)
    drow .*= ws.rscale
    dcol .*= ws.cscale
    return ws.storage
end

function ruiz_step_converged(ws::RuizWorkspace; eps = DEFAULT_EPS)
    return norms_converged(ws.rscale; eps = eps / 2) && norms_converged(ws.cscale; eps = eps / 2)
end

function ruiz_converged(ws::RuizWorkspace; eps = DEFAULT_EPS)
    row_col_maxabs!(ws.rownrm, ws.colnrm, ws.storage)
    return norms_converged(ws.rownrm; eps) && norms_converged(ws.colnrm; eps)
end
