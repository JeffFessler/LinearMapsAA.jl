# kron.jl
# test

using LinearMapsAA: LinearMapAA, LinearMapAM, LinearMapAO
using LinearAlgebra: I
using Test: @test, @testset


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

LinearMapAA_test_kron()
