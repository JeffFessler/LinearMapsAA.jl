#=
kron.jl
Kronecker product
2018-01-19, Jeff Fessler, University of Michigan
=#

# export LinearMapAA_test_kron # testing

using Test: @test
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


# test

# kron test
function LinearMapAA_test_kron( ;
    M1 = rand(ComplexF64, 3, 3),
    M2 = rand(ComplexF64, 2, 2),
)

    M12 = kron(M1, M2)
    A1 = LinearMapAA(M1)
    A2 = LinearMapAA(M2)
    @testset "kron basics" begin
        @test kron(A1, A1) isa LinearMapAM
        for pair in ((A1,A2), (A1,M2), (M1,A2)) # all combinations
            AAk = kron(pair[1], pair[2])
            @test AAk isa LinearMapAM
            @test Matrix(AAk) == M12
            @test Matrix(AAk)' == Matrix(AAk')
        end
    end

    A3 = LinearMapAA(M1 ; odim=(1,size(M1,1)))
    A4 = LinearMapAA(M2 ; idim=(size(M2,2),1))
    @testset "kron(I,A)" begin
        K = kron(I(2), A4)
        M = kron(I(2), M2)
        @test K isa LinearMapAO
           @test Matrix(K) == M
           @test Matrix(K') == Matrix(K)'
    end

    @testset "kron(A,I)" begin
        K = kron(A4, I(2))
        M = kron(M2, I(2))
        @test K isa LinearMapAO
           @test Matrix(K) == M
           @test Matrix(K') == Matrix(K)'
    end

    @testset "kron(Ao,M)" begin
        @test kron(A3, M2) isa LinearMapAO
        @test kron(M1, A4) isa LinearMapAO
    end

    @testset "kron(Ao,Bo)" begin
        K = kron(A3, A4)
        M = kron(M1, M2)
        @test K isa LinearMapAO
           @test Matrix(K) == M
           @test Matrix(K') == Matrix(K)'
    end

    true
end
