# ambiguity.jl
# tests needed only because of code added to resolve method ambiguities

using LinearAlgebra: Diagonal, UpperTriangular
using LinearAlgebra: Adjoint, Transpose, Symmetric
using LinearAlgebra: TransposeAbsVec, AdjointAbsVec
using LinearAlgebra: givens
import LinearAlgebra
using LinearMapsAA
using Test: @test, @testset, @test_throws


M = rand(2,2)
A = LinearMapAA(M)

@testset "Diagonal" begin
    D = Diagonal(1:2)
    @test Matrix(A * D) == M * D
    @test Matrix(D * A) == D * M
end

@testset "UpperTri" begin
    U = UpperTriangular(M)
    @test Matrix(A * U) ≈ M * U
    @test Matrix(U * A) ≈ U * M
end

@testset "TransposeVec" begin
    xt = Transpose(1:2)
    @test xt isa TransposeAbsVec
    @test xt * M ≈ xt * A
end

@testset "AdjointVec" begin
    r = Adjoint(1:2)
    @test r isa AdjointAbsVec
    @test r * M ≈ r * A
end

@testset "Transpose" begin
# work-around per
# https://github.com/Jutho/LinearMaps.jl/issues/147
    LinearAlgebra.isposdef(A::Transpose) = LinearAlgebra.isposdef(parent(A))
    S = LinearAlgebra.Symmetric(M)
    T = LinearAlgebra.Transpose(S)
    @test Matrix(A * T) ≈ M * T # failed prior to isposdef overload
    @test Matrix(T * A) ≈ T * M
end

@testset "Adjoint" begin
    C = rand(ComplexF32, 2, 2)
    H = LinearAlgebra.Hermitian(C)
    J = Adjoint(H)
    LinearAlgebra.isposdef(A::Adjoint) = LinearAlgebra.isposdef(parent(A))
    @test Matrix(A * J) ≈ M * J
    @test Matrix(J * A) ≈ J * M
end

@testset "AbsRot" begin
    G, _ = givens(1., 2., 3, 4)
    R = LinearAlgebra.Rotation([G, G])
    AR = adjoint(R); # semicolon needed, otherwise "show" error!
    # size(AR) # fails because AR is size-less and not an AbstractMatrix
    AR isa Adjoint{<:Any,<:LinearAlgebra.AbstractRotation} # true
    # ones(4,4) * AR # works
    @test_throws String A * AR
end
