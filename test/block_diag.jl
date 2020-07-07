# block_diag.jl
# test

using LinearMapsAA: LinearMapAA, LinearMapAM, LinearMapAO
using SparseArrays: blockdiag, sparse
using Test: @test, @testset


M1 = rand(3,2)
M2 = rand(5,4)
M = [M1 zeros(size(M1,1),size(M2,2));
	zeros(size(M2,1),size(M1,2)) M2]

A1 = LinearMapAA(M1) # WrappedMap
A2 = LinearMapAA(x -> M2*x, y -> M2'y, size(M2), T=eltype(M2)) # FunctionMap
B = block_diag(A1, A2)

@testset "block_diag for AM" begin
	@test Matrix(B) == M
	@test Matrix(B)' == Matrix(B')
end

# test LMAO
@testset "block_diag for AO" begin
	Ao = LinearMapAA(M2 ; odim=(1,5))
	@test block_diag(A1, A1) isa LinearMapAM
	@test block_diag(A1, A1 ; tryop=true) isa LinearMapAO
	@test block_diag(A1, Ao) isa LinearMapAM
	Bo = block_diag(Ao, Ao)
	@test Bo isa LinearMapAO

	Md = blockdiag(sparse(M2), sparse(M2))
	X = rand(4,2)
	Yo = Bo * X
	Yd = Md * vec(X)
	@test vec(Yo) ≈ Yd

	@test Matrix(Bo)' == Matrix(Bo')
end
