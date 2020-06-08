#=
block_diag.jl
Like SparseArrays:blockdiag but slightly different name, just to be safe,
because SparseArrays are also an AbstractArray type.

Motivated by compressed sensing dynamic MRI
where each frame has different k-space sampling
2020-06-08 Jeff Fessler
=#

export block_diag
import LinearMaps


"""
    B = block_diag(As::LinearMapAA...)
"""
function block_diag(As::LinearMapAA...)
	B = LinearMaps.blockdiag(map(A -> A._lmap, As)...)
	prop = (nblock = length(As),)
	return LinearMapAA(B, prop)
end


"""
    block_diag(:test)
self test
"""
function block_diag(test::Symbol)
	!(test === :test) && throw("error $test")

	M1 = rand(4,3)
	M2 = rand(5,6)
	M = [M1 zeros(size(M1,1),size(M2,2));
		zeros(size(M2,1),size(M1,2)) M2]

	A1 = LinearMapAA(M1)
	A2 = LinearMapAA(x -> M2*x, y -> M2'y, size(M2))
	B = block_diag(A1, A2)

	@test Matrix(B) == M
	@test Matrix(B)' == Matrix(B')

	true
end

block_diag(:test)
