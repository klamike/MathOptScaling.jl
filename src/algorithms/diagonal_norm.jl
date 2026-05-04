struct DiagonalScaling{R,C}
    row::R
    col::C
end

struct DiagonalNormWorkspace{S,W}
    scaling::S
    storage::W
end

function DiagonalNormWorkspace(A)
    drow, dcol = scaling_vectors(A)
    return DiagonalNormWorkspace(DiagonalScaling(drow, dcol), A)
end

valid_p(p::Real) = p == Inf || (isfinite(p) && p > 0)
valid_p(::Val{P}) where {P} = P === :inf || valid_p(P)
valid_p(_) = false
check_p(p) = valid_p(p) ? nothing : throw(ArgumentError("p must be positive or Inf"))

row_col_p_norms!(rowdst, coldst, A, ::Val{:inf}) = row_col_maxabs!(rowdst, coldst, A)
row_col_p_norms!(rowdst, coldst, A, ::Val{2}) = row_col_norm2!(rowdst, coldst, A)

function row_col_p_norms!(rowdst, coldst, A, p)
    row_col_sum_abs_power!(rowdst, coldst, A, p)
    pow_inv!(rowdst, p)
    pow_inv!(coldst, p)
    return rowdst, coldst
end

diagonal_norm_scaling(A; kwargs...) = diagonal_norm_scaling!(copy(A); kwargs...)

diagonal_norm_scaling!(A; p = Val(:inf)) =
    diagonal_norm_scaling!(A, DiagonalNormWorkspace(A); p)

function diagonal_norm_scaling!(A, ws::DiagonalNormWorkspace; p = Val(:inf))
    p = pow_arg(eltype(A), p)
    check_p(p)
    drow, dcol = ws.scaling.row, ws.scaling.col
    row_col_p_norms!(drow, dcol, ws.storage, p)
    inv_sqrt!(drow, drow)
    inv_sqrt!(dcol, dcol)
    scale_rows_cols!(ws.storage, drow, dcol)
    return ws.storage, ws.scaling
end
