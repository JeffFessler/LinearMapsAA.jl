#=
types.jl
Types and constructors for LinearMapAA objects.
2026-06-27 Jeff Fessler, University of Michigan
=#

using LinearMaps: LinearMap

LMAAkeys = (:_lmap, :_prop, :_idim, :_odim) # reserved


#=
Note: this old way may not properly allow `setindex!` to work as desired
because it may change the type of the lmap and of the prop:
`struct LinearMapAA{T, M <: LinearMap, P <: NamedTuple} <: AbstractMatrix{T}`
{T, M <: LinearMap, P <: NamedTuple}
    _lmap::M
    _prop::P
    function LinearMapAA{T}(L::M, p::P) where {T, M <: LinearMap, P <: NamedTuple}
    function LinearMapAA(L::LinearMap, p::NamedTuple) # where {T, M <: LinearMap, P <: NamedTuple}
   #    new{T,M,P}(L, p)
        new(L, p)
    end
=#


"""
    mutable struct LinearMapAM{T,Do,Di} <: AbstractMatrix{T}
"matrix" version that is quite akin to a matrix in its behavior
"""
mutable struct LinearMapAM{T,Do,Di} <: AbstractMatrix{T}
    _lmap::LinearMap # "L"
    _prop::NamedTuple # user-defined "named properties" accessible via A.name
    _idim::Dims{Di} # "input" dimensions, always (size(L,2),)
    _odim::Dims{Do} # "output" dimensions, always (size(L,1),)
end


"""
    struct LinearMapAO{T,Do,Di}
"tensor" version that can map from arrays to arrays
(it is not a subtype of `AbstractArray` and `setindex` is unsupported)
"""
struct LinearMapAO{T,Do,Di}
    _lmap::LinearMap # "L"
    _prop::NamedTuple # user-defined "named properties" accessible via A.name
    _idim::Dims{Di} # "input" dimensions, often (size(L,2),)
    _odim::Dims{Do} # "output" dimensions, often (size(L,1),)
end


# most operations apply to both AM and AO types:
LinearMapAX{T,Do,Di} =
    Union{ LinearMapAM{T,Do,Di}, LinearMapAO{T,Do,Di} } where {T,Do,Di}


# constructors


"""
    B = LinearMapAO{T,Do,Di}(A::LinearMapAX)
Make an AO from an AM, despite `idim` and `odim` being 1D,
for expert users who want `B*X` to be an `Array`.
Somewhat an opposite of `undim`.
"""
LinearMapAO(A::LinearMapAX{T,Do,Di}) where {T,Do,Di} =
    LinearMapAO{T,Do,Di}(A._lmap, A._prop, A._idim, A._odim)


"""
    A = LinearMapAA(L::LinearMap ; ...)

Constructor

options
- `prop::NamedTuple = NamedTuple()`
- `T = eltype(L)`
- `idim::Dims = (size(L,2),)`
- `odim::Dims = (size(L,1),)`
- `operator::Bool` by default: `false` if both `idim` & `odim` are 1D.

`prop` cannot include the fields `_lmap`, `_prop`, `_idim`, `_odim`

Output `A` is `LinearMapAO` if `operator` is `true`, else `LinearMapAM`.
"""
function LinearMapAA(L::LinearMap ;
    prop::NamedTuple = NamedTuple(),
    T::Type = eltype(L),
    idim::Dims{Di} = (size(L,2),),
    odim::Dims{Do} = (size(L,1),),
    operator::Bool = length(idim) > 1 || length(odim) > 1,
) where {Di,Do}

    size(L,2) == prod(idim) ||
        throw(DimensionMismatch("size2=$(size(L,2)) vs idim=$idim"))
    size(L,1) == prod(odim) ||
        throw(DimensionMismatch("size1=$(size(L,1)) vs odim=$odim"))
    length(intersect(propertynames(prop), LMAAkeys)) > 0 &&
        throw("invalid property field among $(propertynames(prop))")

    return operator ?
         LinearMapAO{T,Do,Di}(L, prop, idim, odim) :
         LinearMapAM{T,Do,Di}(L, prop, idim, odim)
end


# for backwards compatibility:
LinearMapAA(L, prop::NamedTuple ; kwargs...) =
    LinearMapAA(L::LinearMap ; prop=prop, kwargs...)


"""
    A = LinearMapAA(L::AbstractMatrix ; ...)
Constructor
"""
LinearMapAA(L::AbstractMatrix ; kwargs...) =
    LinearMapAA(LinearMap(L) ; kwargs...)


_ismutating(f) = first(methods(f)).nargs == 3


"""
    A = LinearMapAA(f::Function, fc::Function, D::Dims{2} [, prop::NamedTuple)]
    ; T::DataType = Float32, idim::Dims, odim::Dims)
Constructor
"""
function LinearMapAA(f::Function, fc::Function, D::Dims{2} ;
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
            x -> reshape(f(reshape(x,idim)),odim)
        fcnew = _ismutating(fc) ?
            (x,y) -> f(reshape(x,idim), reshape(y,idim)) :
            y -> reshape(fc(reshape(y,odim)),idim)
    end
    LinearMapAA(LinearMap{T}(fnew, fcnew, D[1], D[2]) ;
        idim=idim, odim=odim, kwargs...)
end

LinearMapAA(f::Function, fc::Function, D::Dims{2}, prop::NamedTuple ; kwargs...) =
    LinearMapAA(f, fc, D ; prop=prop, kwargs...)


"""
    A = LinearMapAA(f::Function, D::Dims{2} [, prop::NamedTuple]; kwargs...)
Constructor
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
            x -> reshape(f(reshape(x,idim)),odim)
    end
    LinearMapAA(LinearMap{T}(fnew, D[1], D[2]) ;
        idim=idim, odim=odim, kwargs...)
end

LinearMapAA(f::Function, D::Dims{2}, prop::NamedTuple ; kwargs...) =
    LinearMapAA(f, D ; prop=prop, kwargs...)
