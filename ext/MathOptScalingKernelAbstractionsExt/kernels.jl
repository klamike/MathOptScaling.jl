@inline atomic_max!(dst, i, value) = begin
    old = dst[i]
    while value > old
        result = Atomix.@atomicreplace dst[i] old => value
        old = result.old
        result.success && break
    end
    return
end

@inline atomic_min_nonzero!(dst, i, value) = begin
    iszero(value) && return
    old = dst[i]
    while value < old
        result = Atomix.@atomicreplace dst[i] old => value
        old = result.old
        result.success && break
    end
    return
end

@inline atomic_hypot!(dst, i, value) = begin
    iszero(value) && return
    old = dst[i]
    while true
        result = Atomix.@atomicreplace dst[i] old => hypot(old, value)
        old = result.old
        result.success && break
    end
    return
end

@inline atomic_combine!(dst, i, x, _, ::Val{:max}) = atomic_max!(dst, i, abs(x))
@inline atomic_combine!(dst, i, x, _, ::Val{:hypot}) = atomic_hypot!(dst, i, x)
@inline atomic_combine!(dst, i, x, p, ::Val{:sum}) = begin
    val = abs_power(x, p)
    iszero(val) || @inbounds Atomix.@atomic dst[i] += val
    return
end

@kernel coo_atomic_reduce_kernel!(rowdst, coldst, @Const(rowval), @Const(colval), @Const(nzval), rowp, colp, op) = begin
    k = @index(Global, Linear)
    @inbounds begin
        x = nzval[k]
        atomic_combine!(rowdst, rowval[k], x, rowp, op)
        atomic_combine!(coldst, colval[k], x, colp, op)
    end
end

@kernel coo_minmax_kernel!(rowmin, rowmax, colmin, colmax, @Const(rowval), @Const(colval), @Const(nzval)) = begin
    k = @index(Global, Linear)
    @inbounds begin
        a = abs(nzval[k])
        if !iszero(a)
            i, j = rowval[k], colval[k]
            atomic_min_nonzero!(rowmin, i, a)
            atomic_max!(rowmax, i, a)
            atomic_min_nonzero!(colmin, j, a)
            atomic_max!(colmax, j, a)
        end
    end
end

@kernel coo_axis_reduce_kernel!(dst, @Const(idx), @Const(nzval), p, op) = begin
    k = @index(Global, Linear)
    @inbounds atomic_combine!(dst, idx[k], nzval[k], p, op)
end

@kernel coo_axis_minmax_kernel!(dstmin, dstmax, @Const(idx), @Const(nzval)) = begin
    k = @index(Global, Linear)
    @inbounds begin
        a = abs(nzval[k])
        if !iszero(a)
            i = idx[k]
            atomic_min_nonzero!(dstmin, i, a)
            atomic_max!(dstmax, i, a)
        end
    end
end
