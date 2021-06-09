# ambiguity.jl
# tests needed only because of code added to resolve method ambiguities

using LinearAlgebra: Diagonal, UpperTriangular #, symmetric
using LinearMapsAA
using Test: @test, @testset

@testset "Diagonal" begin
	M = rand(3,3)
	A = LinearMapAA(M)

	D = Diagonal(1:3)
	@test Matrix(A * D) == M * D
	@test Matrix(D * A) == D * M

	U = UpperTriangular(M)
	@test Matrix(A * U) == M * U
	@test Matrix(U * A) == U * M
end

#=
	C = rand(ComplexF32, 3, 3)
	H = LinearAlgebra.Hermitian(C)
	J = Adjoint(H)

#	@test Matrix(A * J) == M * J # todo
#	@test Matrix(J * A) == J * M

	S = LinearAlgebra.Symmetric(M)
	T = LinearAlgebra.Transpose(S)
	M * T
#	A * T # fails - todo
# https://github.com/Jutho/LinearMaps.jl/issues/147
=#
