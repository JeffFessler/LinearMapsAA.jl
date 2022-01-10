#=
types.jl
Types and constructors for LinearMapAA objects.
2026-06-27 Jeff Fessler, University of Michigan
=#

using LinearMaps: LinearMap

LMAAkeys = (:_lmap, :_prop, :_idim, :_odim) # reserved


"""
    struct LinearMapAM{T,Do,Di,LM,P} <: AbstractMatrix{T}
"matrix" version that is quite akin to a matrix in its behavior
"""
struct LinearMapAM{T,Do,Di,LM,P} <: AbstractMatrix{T}
    _lmap::LM # "L"
    _prop::P # user-defined "named properties" accessible via A.name
    _idim::Dims{Di} # "input" dimensions, always (size(L,2),)
    _odim::Dims{Do} # "output" dimensions, always (size(L,1),)
end


"""
    struct LinearMapAO{T,Do,Di,LM,P}
"Tensor" version that can map from arrays to arrays.
(It is not a subtype of `AbstractArray`.)
"""
struct LinearMapAO{T,Do,Di,LM,P} # LM <: LinearMap, P <: NamedTuple
    _lmap::LM # "L"
    _prop::P # user-defined "named properties" accessible via A.name
    _idim::Dims{Di} # "input" dimensions, often (size(L,2),)
    _odim::Dims{Do} # "output" dimensions, often (size(L,1),)
end


"""
    struct LinearMapAX{T,Do,Di,LM,P}
Union of `LinearMapAM` and `LinearMapAO`
because most operations apply to both AM and AO types.
* `T` : `eltype`
* `Do` : output dimensionality
* `Di` : input dimensionality
* `LM` : `LinearMap` type
* `P` : `NamedTuple` type
"""
LinearMapAX{T,Do,Di,LM,P} =
    Union{
        LinearMapAM{T,Do,Di,LM,P},
        LinearMapAO{T,Do,Di,LM,P},
    } where {T,Do,Di,LM,P}


# constructors


"""
    B = LinearMapAO(A::LinearMapAX)
Make an AO from an AM, despite `idim` and `odim` being 1D,
for expert users who want `B*X` to be an `Array`.
Somewhat an opposite of `undim`.
"""
LinearMapAO(A::LinearMapAX{T,Do,Di,LM,P}) where {T,Do,Di,LM,P} =
    LinearMapAO{T,Do,Di,LM,P}(A._lmap, A._prop, A._idim, A._odim)


"""
    A = LinearMapAA(L::LinearMap ; ...)

Constructor for `LinearMapAM`  or `LinearMapAO` given a `LinearMap`.

# Options
- `prop::NamedTuple = NamedTuple()`;
  cannot include the fields `_lmap`, `_prop`, `_idim`, `_odim`
- `T = eltype(L)`
- `idim::Dims = (size(L,2),)`
- `odim::Dims = (size(L,1),)`
- `operator::Bool` by default: `false` if both `idim` & `odim` are 1D.

Output `A` is `LinearMapAO` if `operator` is `true`, else `LinearMapAM`.
"""
function LinearMapAA(
    L::LM ;
    prop::P = NamedTuple(),
    T::Type = eltype(L),
    idim::Dims{Di} = (size(L,2),),
    odim::Dims{Do} = (size(L,1),),
    operator::Bool = length(idim) > 1 || length(odim) > 1,
) where {Di, Do, LM <: LinearMap, P <: NamedTuple}

    size(L,2) == prod(idim) ||
        throw(DimensionMismatch("size2=$(size(L,2)) vs idim=$idim"))
    size(L,1) == prod(odim) ||
        throw(DimensionMismatch("size1=$(size(L,1)) vs odim=$odim"))
    length(intersect(propertynames(prop), LMAAkeys)) > 0 &&
        throw("invalid property field among $(propertynames(prop))")

    return operator ?
         LinearMapAO{T,Do,Di,LM,P}(L, prop, idim, odim) :
         LinearMapAM{T,Do,Di,LM,P}(L, prop, idim, odim)
end


# for backwards compatibility:
LinearMapAA(L, prop::NamedTuple ; kwargs...) =
    LinearMapAA(L::LinearMap ; prop, kwargs...)


"""
    A = LinearMapAA(L::AbstractMatrix ; ...)
Constructor given an `AbstractMatrix`.
"""
LinearMapAA(L::AbstractMatrix ; kwargs...) =
    LinearMapAA(LinearMap(L) ; kwargs...)


_ismutating(f) = first(methods(f)).nargs == 3


"""
    A = LinearMapAA(f::Function, fc::Function, D::Dims{2} [, prop::NamedTuple)]
    ; T::DataType = Float32, idim::Dims, odim::Dims)
Constructor given forward `f` and adjoint function `fc`.
"""
function LinearMapAA(
    f::Function,
    fc::Function,
    D::Dims{2} ;
    T::DataType = Float32,
    idim::Dims = (D[2],),
    odim::Dims = (D[1],),
    kwargs...,
)

    if idim == (D[2],) && odim == (D[1],)
        fnew = f
        fcnew = fc
    else
        fnew = _ismutating(f) ?
            (y,x) -> f(reshape(y,odim), reshape(x,idim)) :
            x -> reshape(f(reshape(x,idim)), odim)
        fcnew = _ismutating(fc) ?
            (x,y) -> fc(reshape(x,idim), reshape(y,odim)) :
            y -> reshape(fc(reshape(y,odim)), idim)
    end
    LinearMapAA(LinearMap{T}(fnew, fcnew, D[1], D[2]) ;
        idim=idim, odim=odim, kwargs...)
end

LinearMapAA(f::Function, fc::Function, D::Dims{2}, prop::NamedTuple ; kwargs...) =
    LinearMapAA(f, fc, D ; prop=prop, kwargs...)


"""
    A = LinearMapAA(f::Function, D::Dims{2} [, prop::NamedTuple]; kwargs...)
Constructor given just forward function `f`.
"""
function LinearMapAA(f::Function, D::Dims{2} ;
    T::DataType = Float32,
    idim::Dims = (D[2],),
    odim::Dims = (D[1],),
    kwargs...,
)

    if idim == (D[2],) && odim == (D[1],)
        fnew = f
    else
        fnew = _ismutating(f) ?
            (y,x) -> f(reshape(y,odim), reshape(x,idim)) :
            x -> reshape(f(reshape(x,idim)), odim)
    end
    LinearMapAA(LinearMap{T}(fnew, D[1], D[2]) ;
        idim=idim, odim=odim, kwargs...)
end

LinearMapAA(f::Function, D::Dims{2}, prop::NamedTuple ; kwargs...) =
    LinearMapAA(f, D ; prop=prop, kwargs...)
