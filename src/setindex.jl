#=
setindex.jl

Provide setindex!() capabilities like A[i,j] = v for LinearMapAA objects.

The setindex! function here is (necessarily) quite inefficient.
It is provided solely so that a LinearMapAA conforms to the AbstractMatrix
requirement that a setindex! method exists.
Use of this function is strongly discouraged, other than for testing.

2018-01-19, Jeff Fessler, University of Michigan
=#

using Test: @test

"""
`A[i,j] = v`

Mathematically, if B = copy(A) and then we say B[i,j] = v
then `B = A + (v - A[i,j]) e_i e_j'`
where `e_i` and `e_j` are the appropriate unit vectors.
Using this basic math, we can perform `B*x` using `A*x` and `A[i,j]`.

This method works because LinearMapAA is a *mutable* struct.
"""
function Base.setindex!(A::LinearMapAA, v::Number, i::Int, j::Int)
	L = A._lmap
	if true
		forw = (x) ->
			begin
				tmp = L * x 
				tmp[i] += (v - L[i,j]) * x[j]
				return tmp
			end
	end

	has_adj = !(typeof(L) <: LinearMaps.FunctionMap) || (L.fc !== nothing)
	if has_adj
		back = (y) ->
			begin
				tmp = L' * y 
				tmp[j] += conj(v - L[i,j]) * y[i]
				return tmp
			end
	end

#=
	# for future reference, some of this could be implemented
	A.issymmetric = false # could check if i==j
	A.ishermitian = false # could check if i==j and v=v'
	A.posdef = false # could check if i==j and v > A[i,j]
=#

	if has_adj
		return LinearMapAA(f, fc, size(L))
	else
		return LinearMapAA(f, size(L))
	end
end


"""
`setindex!(A::LinearMapAA, v::Number, k::Int)`

Provide the single index version to meet the `AbstractArray` spec:
https://docs.julialang.org/en/latest/manual/interfaces/#Indexing-1
"""
function Base.setindex!(A::LinearMapAA, v::Number, k::Int)
	c = CartesianIndices(size(A))[k] # is there a more elegant way?
	setindex!(A, v::Number, c[1], c[2])
end


"""
`LinearMapAA_test_setindex(A::LinearMapAA)`
"""
function LinearMapAA_test_setindex(A::LinearMapAA)

    @test all(size(A) .>= (4,4)) # required by tests

	# A[i,j]
	B = copy(A)
	(i,j) = (2,3)
	v = 1 + A[i,j]^2 # this value must differ from A[i,j]
	B[i,j] = v
	Am = Matrix(A)
	Bm = Matrix(B)
	Am[i,j] = v
	@test isapprox(Am, Bm)

	# A[i]
	B = copy(A)
	i = 5
	v = 2 + A[i]^2 # this value must differ from A[i]
	B[i] = v
	Am = Matrix(A)
	Bm = Matrix(B)
	Am[i] = v
	@test isapprox(Am, Bm)

	# A[:,j]
	B = copy(A)
	j = 3
	v = 2 .+ A[:,j].^2
	B[:,j] .= v
	Am = Matrix(A)
	Bm = Matrix(B)
	Am[:,j] .= v
	@test isapprox(Am, Bm)

	# insanity below here

	# A[:]
	B = copy(A)
	B[:] .= 5
	Am = Matrix(A)
	Am[:] .= 5
	Bm = Matrix(B)
	@test Bm == Am

	# A[:,:]
	B = copy(A)
	B[:,:] .= 6
	Am = Matrix(A)
	Am[:,:] .= 6
	Bm = Matrix(B)
	@test Bm == Am

	true
end
