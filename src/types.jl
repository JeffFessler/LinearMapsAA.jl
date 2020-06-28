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
    A = LinearMapAA(L::LinearMap ; ...)

Constructor

options
- `prop::NamedTuple = NamedTuple()`
- `T = eltype(L)`
- `idim::Dims = (size(L,2),)`
- `odim::Dims = (size(L,1),)`

`prop` cannot include the fields `_lmap`, `_prop`, `_idim`, `_odim`
"""
function LinearMapAA(L::LinearMap ;
    prop::NamedTuple = NamedTuple(),
    T::Type = eltype(L),
    idim::Dims{Di} = (size(L,2),),
    odim::Dims{Do} = (size(L,1),),
) where {Di,Do}

    size(L,2) == prod(idim) ||
        throw(DimensionMismatch("size2=$(size(L,2)) vs idim=$idim"))
    size(L,1) == prod(odim) ||
        throw(DimensionMismatch("size1=$(size(L,1)) vs odim=$odim"))
    length(intersect(propertynames(prop), LMAAkeys)) > 0 &&
        throw("invalid property field among $(propertynames(prop))")

    return ((idim == (size(L,2),)) && (odim == (size(L,1),))) ?
         LinearMapAM{T,Do,Di}(L, prop, idim, odim) :
         LinearMapAO{T,Do,Di}(L, prop, idim, odim)
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


"""
    A = LinearMapAA(f::Function, fc::Function, D::Dims{2} [, prop::NamedTuple)]
    ; T::DataType = Float32, idim::Dims, odim::Dims)
Constructor
"""
LinearMapAA(f::Function, fc::Function, D::Dims{2} ;
    T::DataType = Float32,
    kwargs...,
) =
    LinearMapAA(LinearMap{T}(f, fc, D[1], D[2]) ; kwargs...)

LinearMapAA(f::Function, fc::Function, D::Dims{2}, prop::NamedTuple ; kwargs...) =
    LinearMapAA(f, fc, D ; prop=prop, kwargs...)


"""
    A = LinearMapAA(f::Function, D::Dims{2} [, prop::NamedTuple]; kwargs...)
Constructor
"""
LinearMapAA(f::Function, D::Dims{2} ; T::DataType = Float32, kwargs...) =
    LinearMapAA(LinearMap{T}(f, D[1], D[2]) ; kwargs...)

LinearMapAA(f::Function, D::Dims{2}, prop::NamedTuple ; kwargs...) =
    LinearMapAA(f, D ; prop=prop, kwargs...)
