# wrap-linop.jl
# Wrap a LinearMapAA around a LinearOperator

export LinearMapAA

using .LinearOperators: LinearOperator


"""
    A = LinearMapAA(L::LinearOperator ; kwargs...)

Wrap a `LinearOperator` in a `LinearMapAX`.
Options are passed to `LinearMapAA` constructor.
"""
function LinearMapAA(L::LinearOperator ; kwargs...)
    forw!(y, x) = mul!(y, L, x)
    back!(x, y) = mul!(x, L', y)
    return LinearMapAA(forw!, back!, size(L); kwargs...)
end
