# identity.jl

import LinearAlgebra: UniformScaling

"""
    *(I, X) = X
    *(J, X) = J.λ * X

Extends the effect of `I::UniformScaling` and scaled versions thereof
to also apply to `X::AbstractArray`
instead of just to `Vector` and `Matrix` types.

This is "type piracy" since `UniformScaling`
does not originate in this package.
Hopefully someday it will be unnecessary; see:
https://github.com/JuliaLang/LinearAlgebra.jl/issues/1656

"""
Base.:(*)(J::UniformScaling, X::AbstractArray) = J.λ == 1 ? X : J.λ * X
