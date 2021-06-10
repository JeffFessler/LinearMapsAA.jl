# ambiguity.jl
# tests needed only because of code added to resolve method ambiguities

using LinearAlgebra: Diagonal, UpperTriangular
using LinearAlgebra: Adjoint, Transpose, Symmetric
using LinearAlgebra: TransposeAbsVec, AdjointAbsVec
import LinearAlgebra
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

@testset "Transpose" begin
# work-around per
# https://github.com/Jutho/LinearMaps.jl/issues/147
    LinearAlgebra.isposdef(A::Transpose) = LinearAlgebra.isposdef(parent(A))
    S = LinearAlgebra.Symmetric(M)
    T = LinearAlgebra.Transpose(S)
    @test Matrix(A * T) == M * T # failed prior to isposdef overload
    @test Matrix(T * A) == T * M
end

@testset "Adjoint" begin
    C = rand(ComplexF32, 3, 3)
    H = LinearAlgebra.Hermitian(C)
    J = Adjoint(H)
    LinearAlgebra.isposdef(A::Adjoint) = LinearAlgebra.isposdef(parent(A))
    @test Matrix(A * J) == M * J
    @test Matrix(J * A) == J * M
end
