struct ChambollePockScaling{R,C,S,T}
    row::R
    col::C
    sigma::S
    tau::T
end

check_alpha(alpha::Real) =
    0 <= alpha <= 2 || throw(ArgumentError("alpha must be in [0, 2]"))
check_alpha(::Val{A}) where {A} =
    A isa Real ? check_alpha(A) : throw(ArgumentError("alpha must be in [0, 2]"))
alpha_complement(::Val{A}) where {A} = Val(2 - A)
alpha_complement(alpha::Real) = 2 - alpha

struct ChambollePockWorkspace{C,W}
    scaling::C
    storage::W
end

function ChambollePockWorkspace(A)
    sigma, tau = scaling_vectors(A)
    row, col = similar(sigma), similar(tau)
    return ChambollePockWorkspace(ChambollePockScaling(row, col, sigma, tau), A)
end

chambolle_pock_scaling(A; kwargs...) = chambolle_pock_scaling!(copy(A); kwargs...)

chambolle_pock_scaling!(A; alpha = Val(1)) =
    chambolle_pock_scaling!(A, ChambollePockWorkspace(A); alpha)

function chambolle_pock_scaling!(A, ws::ChambollePockWorkspace; alpha = Val(1))
    alpha = pow_arg(eltype(A), alpha)
    check_alpha(alpha)
    return _chambolle_pock_scaling!(ws, alpha)
end

function _chambolle_pock_scaling!(ws::ChambollePockWorkspace, ::Val{0})
    scaling = ws.scaling
    row_norm2!(scaling.sigma, ws.storage)
    safe_inv!(scaling.row, scaling.sigma)
    col_sum_abs_power!(scaling.tau, ws.storage, Val(0))
    safe_inv!(scaling.tau, scaling.tau)
    scaling.col .= sqrt.(scaling.tau)
    scaling.sigma .= abs2.(scaling.row)
    return finish_chambolle_pock_scaling!(ws)
end

function _chambolle_pock_scaling!(ws::ChambollePockWorkspace, ::Val{2})
    scaling = ws.scaling
    col_norm2!(scaling.tau, ws.storage)
    safe_inv!(scaling.col, scaling.tau)
    row_sum_abs_power!(scaling.sigma, ws.storage, Val(0))
    safe_inv!(scaling.sigma, scaling.sigma)
    scaling.row .= sqrt.(scaling.sigma)
    scaling.tau .= abs2.(scaling.col)
    return finish_chambolle_pock_scaling!(ws)
end

function _chambolle_pock_scaling!(ws::ChambollePockWorkspace, alpha)
    scaling = ws.scaling
    row_col_sum_abs_power!(scaling.sigma, scaling.tau, ws.storage, alpha_complement(alpha), alpha)
    safe_inv!(scaling.sigma, scaling.sigma)
    safe_inv!(scaling.tau, scaling.tau)
    scaling.row .= sqrt.(scaling.sigma)
    scaling.col .= sqrt.(scaling.tau)
    return finish_chambolle_pock_scaling!(ws)
end

function finish_chambolle_pock_scaling!(ws::ChambollePockWorkspace)
    scale_rows_cols!(ws.storage, ws.scaling.row, ws.scaling.col)
    return ws.storage, ws.scaling
end