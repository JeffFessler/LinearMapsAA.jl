# ambiguity.jl
# tests needed only because of code added to resolve method ambiguities

using LinearAlgebra: Diagonal, UpperTriangular
using LinearAlgebra: Adjoint, Transpose #, symmetric
using LinearAlgebra: TransposeAbsVec, AdjointAbsVec
using LinearMapsAA
using Test: @test, @testset

M = rand(3,3)
A = LinearMapAA(M)

@testset "Diagonal" begin
    D = Diagonal(1:3)
    @test Matrix(A * D) == M * D
    @test Matrix(D * A) == D * M
end

@testset "UpperTri" begin
    U = UpperTriangular(M)
    @test Matrix(A * U) == M * U
    @test Matrix(U * A) == U * M
end

@testset "TransposeVec" begin
    xt = Transpose(1:3)
    @test xt isa TransposeAbsVec
    @test xt * M == xt * A
end

@testset "AdjointVec" begin
    r = Adjoint(1:3)
    @test r isa AdjointAbsVec
    @test r * M == r * A
end

#=
    C = rand(ComplexF32, 3, 3)
    H = LinearAlgebra.Hermitian(C)
    J = Adjoint(H)

#   @test Matrix(A * J) == M * J # todo
#   @test Matrix(J * A) == J * M

    S = LinearAlgebra.Symmetric(M)
    T = LinearAlgebra.Transpose(S)
    M * T
#   A * T # fails - todo
# https://github.com/Jutho/LinearMaps.jl/issues/147
=#
