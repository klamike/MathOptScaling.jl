struct GeometricMeanScaling{R,C}
    row::R
    col::C
end

struct GeometricMeanWorkspace{S,R,C,W}
    scaling::S
    rscale::R
    cscale::C
    rowmin::R
    rowmax::R
    colmin::C
    colmax::C
    storage::W
end

geometric_mean_scaling(A; kwargs...) = geometric_mean_scaling!(copy(A); kwargs...)

function GeometricMeanWorkspace(A)
    drow, dcol = scaling_vectors(A)
    return GeometricMeanWorkspace(
        GeometricMeanScaling(drow, dcol),
        similar(drow), similar(dcol),
        similar(drow), similar(drow),
        similar(dcol), similar(dcol),
        A,
    )
end

geometric_mean_scaling!(A; kwargs...) =
    geometric_mean_scaling!(A, GeometricMeanWorkspace(A); kwargs...)

geometric_mean_scaling!(A, ws::GeometricMeanWorkspace; kwargs...) =
    run_iterative_scaling!(ws; step! = geometric_mean_step!,
        check_converged! = geometric_mean_step_converged, converged! = geometric_mean_converged, kwargs...)

function geometric_mean_step!(ws::GeometricMeanWorkspace)
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

function geometric_mean_step_converged(ws::GeometricMeanWorkspace; eps = DEFAULT_EPS)
    return norms_converged(ws.rscale; eps) && norms_converged(ws.cscale; eps)
end

function geometric_mean_converged(ws::GeometricMeanWorkspace; eps = DEFAULT_EPS)
    row_col_minmaxabs!(ws.rowmin, ws.rowmax, ws.colmin, ws.colmax, ws.storage)
    geometric_factors!(ws.rscale, ws.rowmin, ws.rowmax)
    geometric_factors!(ws.cscale, ws.colmin, ws.colmax)
    return geometric_mean_step_converged(ws; eps)
end
