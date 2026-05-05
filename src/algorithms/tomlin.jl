"""
Tomlin (1975) iterative scaling: at each step, rescale each row by `1/√(min · max)`
of its absolute nonzeros, then do the same for columns. Often called "geometric mean
scaling" in solver literature, where the geometric mean is taken over (min, max) only,
not over all entries.
"""

struct TomlinScaling{R,C}
    row::R
    col::C
end

struct TomlinWorkspace{S,R,C,W}
    scaling::S
    rscale::R
    cscale::C
    rowmin::R
    rowmax::R
    colmin::C
    colmax::C
    storage::W
end

tomlin_scaling(A; kwargs...) = tomlin_scaling!(copy(A); kwargs...)

function TomlinWorkspace(A)
    drow, dcol = scaling_vectors(A)
    return TomlinWorkspace(
        TomlinScaling(drow, dcol),
        similar(drow), similar(dcol),
        similar(drow), similar(drow),
        similar(dcol), similar(dcol),
        A,
    )
end

tomlin_scaling!(A; kwargs...) =
    tomlin_scaling!(A, TomlinWorkspace(A); kwargs...)

tomlin_scaling!(A, ws::TomlinWorkspace; kwargs...) =
    run_iterative_scaling!(ws; step! = tomlin_step!,
        check_converged! = tomlin_step_converged, converged! = tomlin_converged, kwargs...)

function tomlin_step!(ws::TomlinWorkspace)
    drow, dcol = ws.scaling.row, ws.scaling.col

    row_minmaxabs!(ws.rowmin, ws.rowmax, ws.colmin, ws.colmax, ws.storage)
    geometric_factors!(ws.rscale, ws.rowmin, ws.rowmax)
    scale_rows!(ws.storage, ws.rscale)
    drow .*= ws.rscale

    col_minmaxabs!(ws.rowmin, ws.rowmax, ws.colmin, ws.colmax, ws.storage)
    geometric_factors!(ws.cscale, ws.colmin, ws.colmax)
    scale_cols!(ws.storage, ws.cscale)
    dcol .*= ws.cscale
    return ws.storage
end

function tomlin_step_converged(ws::TomlinWorkspace; eps = DEFAULT_EPS)
    return norms_converged(ws.rscale; eps) && norms_converged(ws.cscale; eps)
end

function tomlin_converged(ws::TomlinWorkspace; eps = DEFAULT_EPS)
    row_col_minmaxabs!(ws.rowmin, ws.rowmax, ws.colmin, ws.colmax, ws.storage)
    geometric_factors!(ws.rscale, ws.rowmin, ws.rowmax)
    geometric_factors!(ws.cscale, ws.colmin, ws.colmax)
    return tomlin_step_converged(ws; eps)
end
