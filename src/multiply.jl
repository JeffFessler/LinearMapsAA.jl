#=
multiply.jl
Multiplication of a LinearMapAX object with other things
2020-06-16 add 5-arg mul!
2020-06-27 Jeff Fessler
=#

export mul!

using LinearMaps
using LinearAlgebra: UniformScaling, I
import LinearAlgebra: Adjoint, Transpose
import LinearAlgebra: AdjointAbsVec, TransposeAbsVec
import LinearAlgebra: mul! #, lmul!, rmul!


# multiply with I or s*I (identity or scaled identity)
Base.:(*)(A::LinearMapAX, B::UniformScaling{Bool}) = B.λ ? A : (A * B.λ)
Base.:(*)(A::LinearMapAX, B::UniformScaling) = (B.λ == 1) ? A : (A * B.λ)
Base.:(*)(B::UniformScaling{Bool}, A::LinearMapAX) = B.λ ? A : (B.λ * A)
Base.:(*)(B::UniformScaling, A::LinearMapAX) = (B.λ == 1) ? A : (B.λ * A)


# multiply with scalars
Base.:(*)(s::Number, A::LinearMapAX) =
    LinearMapAA((s*I) * A._lmap ; prop=A._prop, idim=A._idim, odim=A._odim,
        operator = isa(A, LinearMapAO))
Base.:(*)(A::LinearMapAX, s::Number) =
    LinearMapAA(A._lmap * (s*I) ; prop=A._prop, idim=A._idim, odim=A._odim,
        operator = isa(A, LinearMapAO))


# multiply LMAX objects, if compatible
Base.:(*)(A::LinearMapAO{Ta,Do,D}, B::LinearMapAO{Tb,D,Di}) where {Ta,Tb,Do,D,Di} =
    lm_obj_mul(A, B)
Base.:(*)(A::LinearMapAO{T,Do,1}, B::LinearMapAM) where {T,Do} = lm_obj_mul(A, B)
Base.:(*)(A::LinearMapAM, B::LinearMapAO{T,1,Di}) where {T,Di} = lm_obj_mul(A, B)
Base.:(*)(A::LinearMapAM, B::LinearMapAM) = lm_obj_mul(A, B)

function lm_obj_mul(A::LinearMapAX, B::LinearMapAX)
    (A._idim != B._odim) && throw("dim mismatch")
    LinearMapAA(A._lmap * B._lmap ;
        prop = (prod=(A._prop,B._prop),),
        idim = B._idim, odim = A._odim,
        operator = isa(A, LinearMapAO) | isa(B, LinearMapAO),
    )
end


# multiply with a matrix
# subtle: AM * Matrix and Matrix * AM yield a new AM
# whereas AO * M and M * AO yield a matrix of numbers!
function Base.:(*)(A::LinearMapAM, B::AbstractMatrix)
    (A._idim == (size(B,1),)) || throw("$(A._idim) * $(size(B,1)) mismatch")
    LinearMapAA(A._lmap * LinearMap(B) ; prop=A._prop, odim=A._odim)
end
function Base.:(*)(A::AbstractMatrix, B::LinearMapAM)
    (B._odim == (size(A,2),)) || throw("$(B._odim) * $(size(A,1)) mismatch")
    LinearMapAA(LinearMap(A) * B._lmap ; prop=B._prop, idim=B._idim)
end

# left multiply by transpose or adjoint vector, y'A = (A'*y)'
Base.:(*)(v::AdjointAbsVec, A::LinearMapAM) = (A' * v')'
Base.:(*)(v::TransposeAbsVec, A::LinearMapAM) =
    transpose(transpose(A) * transpose(v))

# if AO has odim=(M,) i.e. Do=1 then
Base.:(*)(v::AdjointAbsVec, A::LinearMapAO{T,1,1}) where {T} = (A' * v')'
#Base.:(*)(v::AdjointAbsVec, A::LinearMapAO{T,1,Di}) where {T,Di} = conj(A' * v')

# todo: matrix? left
#Base.:(*)(A::Adjoint, B::LinearMapAM) = (B' * A')'
#Base.:(*)(A::Transpose, B::LinearMapAM) = transpose(transpose(B) * transpose(A))


# LMAM case is easy!

# multiply with vectors (in-place)
# pass to lmam_mul! to handle composite maps (products) effectively
# 5-arg mul! requires julia 1.3 or later
# mul!(y, A, x, α, β) ≡ y .= A*(α*x) + β*y

mul!(y::AbstractVector, A::LinearMapAM, x::AbstractVector) =
    lm_mul!(y, A._lmap, x, 1, 0)
#    mul!(y, A._lmap, x)

mul!(y::AbstractVector, A::LinearMapAM, x::AbstractVector,
    α::Number, β::Number) =
    lm_mul!(y, A._lmap, x, α, β)
#    mul!(y, A._lmap, x, α, β)

# treat LinearMaps.CompositeMap as special case for in-place operations
function lm_mul!(y::AbstractVector, Lm::LinearMaps.CompositeMap,
    x::AbstractVector, α::Number, β::Number)
    LinearMaps.mul!(y, Lm, x, α, β) # todo: composite buffer
end

# 5-arg mul! for any other type
lm_mul!(y::AbstractVector, Lm::LinearMap,
    x::AbstractVector, α::Number, β::Number) =
    LinearMaps.mul!(y, Lm, x, α, β)

# with array
#=
these are unused because AM * array becomes a new AM
mul!(Y::AbstractArray, A::LinearMapAM, X::AbstractArray, α::Number, β::Number) =
    lmao_mul!(Y, A._lmap, X, α, β ; idim=A._idim, odim=A._odim)

mul!(Y::AbstractArray, A::LinearMapAM, X::AbstractArray) =
    LinearMapsAA.mul!(Y, A, X, 1, 0)
=#

# left mul!
mul!(x::AdjointAbsVec, y::AdjointAbsVec, A::LinearMapAM,
    α::Number, β::Number) = mul!(x, y, A._lmap, α, β)
mul!(x::TransposeAbsVec, y::TransposeAbsVec, A::LinearMapAM,
    α::Number, β::Number) = mul!(x, y, A._lmap, α, β)

# fallback
mul!(x, A::LinearMapAX, y) = throw("unsupported")
mul!(x, y, A::LinearMapAX) = throw("unsupported")


# LMAO case


# 3-arg O*X
mul!(Y::AbstractArray, A::LinearMapAO, X::AbstractArray) =
    mul!(Y, A, X, 1, 0)

# 3-arg X*O left
mul!(Y::AbstractArray, X::AbstractArray, A::LinearMapAO) =
    mul!(Y, X, A, 1, 0)

# 5-arg O*X
mul!(Y::AbstractArray, A::LinearMapAO, X::AbstractArray, α::Number, β::Number) =
    lmao_mul!(Y, A._lmap, X, α, β ; idim=A._idim, odim=A._odim)

# 5-arg X*O left (todo: test complex case)
mul!(Y::AbstractArray, X::AbstractArray, A::LinearMapAO, α::Number, β::Number) =
    lmao_mul!(Y, A._lmap', X, α, β ; idim=A._odim, odim=A._idim) # note!

# left vector y'O
mul!(x::AdjointAbsVec, y::AdjointAbsVec, A::LinearMapAO{T,1,1},
    α::Number, β::Number) where {T,Di} = mul!(x, y, A._lmap, α, β)
mul!(x::TransposeAbsVec, y::TransposeAbsVec, A::LinearMapAO{T,1,1},
    α::Number, β::Number) where {T,Di} = mul!(x, y, A._lmap, α, β)


"""
     lmao_mul!(Y, A, X, α, β ; idim, odim)

Core routine for 5-arg multiply.
If `A._idim = (2,3,4)` and `A._odim = (5,6)` and
if input `X` has size `(2,3,4, 7,8)`
then output `Y` will have size `(5,6, 7,8)`
"""
function lmao_mul!(Y::AbstractArray, Lm::LinearMap, X::AbstractArray,
    α::Number, β::Number ;
    idim = (size(Lm,2),),
    odim = (size(Lm,1),),
)

    Di = length(idim)
    Do = length(odim)
    (Di > ndims(X) || (idim != size(X)[1:Di])) &&
        throw("idim=$(idim) vs size(RHS)=$(size(X))")
    (Do > ndims(Y) || (odim != size(Y)[1:Do])) &&
        throw("odim=$(odim) vs size(LHS)=$(size(Y))")
    size(X)[(Di+1):end] == size(Y)[(Do+1):end] ||
        throw("size(LHS)=$(size(Y)) vs size(RHS)=$(size(X))")

    x = reshape(X, prod(idim), :)
    y = reshape(Y, prod(odim), :)
    K = size(x,2)
    size(y,2) == K || throw("mismatch $(size(y,2)) K=$K")

    for k=1:K
        xk = selectdim(x,2,k)
        yk = selectdim(y,2,k)
        lm_mul!(yk, Lm, xk, α, β)
    end
    return Y
end


# multiply by array, with allocation

# right
Base.:(*)(A::LinearMapAO, X::AbstractArray) = lmax_mul(A, X)
# left
Base.:(*)(X::AbstractArray, A::LinearMapAO) = lmax_mul(A', X) # note!

#=
# this next line caused ambiguous method errors:
# Base.:(*)(A::LinearMapAM, X::AbstractArray) = lmax_mul(A, X)

# so i resort to this awful kludge:
Base.:(*)(A::LinearMapAM, X::AbstractArray{T,4}) where {T} = lmax_mul(A, X)

nah, too difficult, so revert to the AM*M returning an object, per above
=#

function lmax_mul(A::LinearMapAX{T}, X::AbstractArray) where {T}
    Di = length(A._idim)
    Do = length(A._odim)
    (Di > ndims(X) || (A._idim != size(X)[1:Di])) &&
         throw("idim=$(A._idim) vs size(RHS)=$(size(X))")
    extra = size(X)[(Di+1):end]
    Ty = promote_type(T, eltype(X))
    Y = Array{T}(undef, A._odim..., extra...) # allocate
    lmao_mul!(Y, A._lmap, X, 1, 0; idim=A._idim, odim=A._odim)
end


# multiply with vector

# O*v
Base.:(*)(A::LinearMapAO{T,Do,1}, v::AbstractVector) where {T,Do} =
    reshape(A._lmap * v, A._odim)
# u'*O (no, use general X*O above because unclear what this would mean)
#Base.:(*)(u::LinearAlgebra.AdjointAbsVec, A::LinearMapAO) =
#    reshape(A._lmap' * u', A._idim)

# A*v
Base.:(*)(A::LinearMapAM, v::AbstractVector) =
    A._lmap * v
# u'*A (nah, not worth it)
#Base.:(*)(u::LinearAlgebra.AdjointAbsVec, A::LinearMapAM) =
#    (A._lmap' * u')'

#= bad kludge
LMAAmanyFromOne = Union{
    LinearMapAA{T,2,1},
    LinearMapAA{T,3,1},
    LinearMapAA{T,4,1},
    } where {T}

Base.:(*)(A::LMAAmanyFromOne, v::AbstractVector{<:Number}) where {T} =
    reshape(A._lmap * v, A._odim)
Base.:(*)(A::LinearMapAA, v::AbstractVector{<:Number}) where {T} =
    reshape(A._lmap * v, A._odim)
=#


#= these are pointless; see multiplication with scalars above
lmul!(s::Number, A::LinearMapAA) = lmul!(s, A._lmap)
rmul!(A::LinearMapAA, s::Number) = rmul!(A._lmap, s)
=#
