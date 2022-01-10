#=
lm-aa
Core methods for LinearMapAA objects.
2019-08-07 Jeff Fessler, University of Michigan
=#

using LinearMaps: LinearMap
using LinearAlgebra: UniformScaling, I
import LinearAlgebra: issymmetric, ishermitian, isposdef
#import LinearAlgebra: mul! #, lmul!, rmul!
import SparseArrays: sparse

export redim, undim


# Matrix
Base.Matrix(A::LinearMapAX) = Matrix(A._lmap)

# ndims
# Base.ndims(A::LinearMapAX) = ndims(A._lmap) # 2 for AbstractMatrix
Base.ndims(A::LinearMapAO) = ndims(A._lmap) # 2


"""
    show(io::IO, A::LinearMapAX)
    show(io::IO, ::MIME"text/plain", A::LinearMapAX)
Pretty printing for `display`
"""
function Base.show(io::IO, A::LinearMapAX) # short version
    print(io, isa(A, LinearMapAM) ? "LinearMapAM" : "LinearMapAO",
        ": $(size(A,1)) × $(size(A,2))")
end

# multi-line version:
function Base.show(io::IO, ::MIME"text/plain", A::LinearMapAX{T,Do,Di}) where {T,Do,Di}
    show(io, A)
    print(io, " odim=$(A._odim) idim=$(A._idim) T=$T Do=$Do Di=$Di")
    (A._prop != NamedTuple()) && (print(io, "\nprop = "); show(io, A._prop))
    print(io, "\nmap = $(A._lmap)\n")
end

# size
Base.size(A::LinearMapAX) = size(A._lmap)
Base.size(A::LinearMapAX, d::Int) = size(A._lmap, d)

"""
    redim(A::LinearMapAX ; idim::Dims=A._idim, odim::Dims=A._odim)

"Reinterpret" the `idim` and `odim` of `A`
"""
function redim(A::LinearMapAX{T} ;
    idim::Dims=A._idim, odim::Dims=A._odim) where {T}

    prod(idim) == prod(A._idim) || throw("incompatible idim")
    prod(odim) == prod(A._odim) || throw("incompatible odim")
    return LinearMapAA(A._lmap ; prop=A._prop, T=T, idim=idim, odim=odim)
end

"""
    undim(A::LinearMapAX)

"Reinterpret" the `idim` and `odim` of `A` to be of AM type
"""
undim(A::LinearMapAX{T}) where {T} =
    LinearMapAA(A._lmap ; prop=A._prop, T=T)


# adjoint
Base.adjoint(A::LinearMapAX) = LinearMapAA(adjoint(A._lmap) ;
    prop=A._prop, idim=A._odim, odim=A._idim, operator=isa(A,LinearMapAO))

# transpose
Base.transpose(A::LinearMapAX) = LinearMapAA(transpose(A._lmap) ;
    prop=A._prop, idim=A._odim, odim=A._idim, operator=isa(A,LinearMapAO))

# eltype
Base.eltype(A::LinearMapAX) = eltype(A._lmap)

# LinearMap algebraic properties
issymmetric(A::LinearMapAX) = issymmetric(A._lmap)
#ishermitian(A::LinearMapAX{<:Real}) = issymmetric(A._lmap)
ishermitian(A::LinearMapAX) = ishermitian(A._lmap)
isposdef(A::LinearMapAX) = isposdef(A._lmap)

# comparison of LinearMapAX objects, sufficient but not necessary
Base.:(==)(A::LinearMapAX, B::LinearMapAX) =
    eltype(A) == eltype(B) &&
        A._lmap == B._lmap && A._prop == B._prop &&
        A._idim == B._idim && A._odim == B._odim


# convert to sparse
sparse(A::LinearMapAX) = sparse(A._lmap)


# add or subtract objects (with compatible idim,odim)
function Base.:(+)(A::LinearMapAX, B::LinearMapAX)
    (A._idim != B._idim) && throw("idim mismatch in +")
    (A._odim != B._odim) && throw("odim mismatch in +")
    LinearMapAA(A._lmap + B._lmap ;
        idim=A._idim, odim=A._odim,
        prop = (sum=nothing,props=(A._prop,B._prop)),
        operator = isa(A, LinearMapAO) & isa(B, LinearMapAO),
    )
end

# Allow LMAA + AM only if Do=Di=1
function Base.:(+)(A::LinearMapAX, B::AbstractMatrix)
    (length(A._idim) != 1 || length(A._odim) != 1) && throw("use redim")
    LinearMapAA(A._lmap + LinearMap(B) ; prop=A._prop,
        operator = isa(A, LinearMapAO),
    )
end

# But allow LMAA + I for any Do,Di
Base.:(+)(A::LinearMapAX, B::UniformScaling) = # A + I -> A + I(N)
    LinearMapAA(A._lmap + B(size(A,2)) ;
        prop = (Aprop=A._prop, Iscale=B(size(A,2))[1]),
        idim = A._idim,
        odim = A._odim,
        operator = isa(A, LinearMapAO),
    )
Base.:(+)(A::AbstractMatrix, B::LinearMapAX) = B + A

Base.:(-)(A::LinearMapAX, B::LinearMapAX) = A + (-1)*B
Base.:(-)(A::LinearMapAX, B::AbstractMatrix) = A + (-1)*B
Base.:(-)(A::AbstractMatrix, B::LinearMapAX) = A + (-1)*B


# A.?
Base.getproperty(A::LinearMapAX, s::Symbol) =
    (s in LMAAkeys) ? getfield(A, s) :
#   s == :m ? size(A._lmap, 1) :
    haskey(A._prop, s) ? getfield(A._prop, s) :
        throw("unknown key $s")

Base.propertynames(A::LinearMapAX) = (propertynames(A._prop)..., LMAAkeys...)
