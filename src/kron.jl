#=
kron.jl
Kronecker product
2018-01-19, Jeff Fessler, University of Michigan
=#

# export LinearMapAA_test_kron # testing

using LinearAlgebra: I, Diagonal

# kron (requires LM 2.6.0)


"""
    kron(A::LinearMapAX, M::AbstractMatrix)
    kron(M::AbstractMatrix, A::LinearMapAX)
    kron(A::LinearMapAX, B::LinearMapAX)
Kronecker products

Returns a `LinearMapAO` with appropriate `idim` and `odim`
if either argument is a `LinearMapAO`
else returns a `LinearMapAM`
"""
Base.kron(A::LinearMapAM, M::AbstractMatrix) =
    LinearMapAA(kron(A._lmap, M), A._prop)

Base.kron(M::AbstractMatrix, A::LinearMapAM) =
    LinearMapAA(kron(M, A._lmap), A._prop)


Base.kron(A::LinearMapAM, D::Diagonal{<: Number}) =
    LinearMapAA(kron(A._lmap, D), A._prop)

Base.kron(D::Diagonal{<: Number}, A::LinearMapAM) =
    LinearMapAA(kron(D, A._lmap), A._prop)


Base.kron(A::LinearMapAM, B::LinearMapAM) =
    LinearMapAA(kron(A._lmap, B._lmap) ;
        prop = (kron=nothing, props=(A._prop, B._prop)),
    )

Base.kron(A::LinearMapAO, B::LinearMapAO) =
    LinearMapAA(kron(A._lmap, B._lmap) ;
        prop = (kron=nothing, props=(A._prop, B._prop)),
        odim = (B._odim..., A._odim...),
        idim = (B._idim..., A._idim...),
    )


Base.kron(M::AbstractMatrix, A::LinearMapAO) =
    LinearMapAA(kron(M, A._lmap) ; prop = A._prop,
        odim = (A._odim..., size(M,1)),
        idim = (A._idim..., size(M,2)),
    )

Base.kron(A::LinearMapAO, M::AbstractMatrix) =
    LinearMapAA(kron(A._lmap, M) ; prop = A._prop,
        odim = (size(M,1), A._odim...),
        idim = (size(M,2), A._idim...),
    )
