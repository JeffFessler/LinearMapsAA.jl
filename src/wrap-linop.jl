# wrap-linop.jl
# Wrap a LinearMapAA around a LinearOperator or vice-versa

using .LinearOperators: LinearOperator
import .LinearOperators: LinearOperator

export LinearMapAA
export LinearOperator


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


"""
    L = LinearOperator(A::LinearMapAX; symmetric, hermitian)

Wrap a `LinearOperator` around a `LinearMapAX`.
The options `symmetric` and `hermitian` are `false` by default.
"""
function LinearOperator(
    A::LinearMapAX;
    symmetric::Bool = false,
    hermitian::Bool = false,
)
    forw!(y, x) = mul!(y, A, x)
    back!(x, y) = mul!(x, A', y)
    return LinearOperator(
        eltype(A),
        size(A)...,
        symmetric, hermitian,
        forw!,
        nothing, # transpose mul!
        back!,
    )
end
