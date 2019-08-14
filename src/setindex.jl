#=
setindex.jl

Provide setindex!() capabilities like A[i,j] = s for LinearMapAA objects.

The setindex! function here is (necessarily) quite inefficient.
It is provided solely so that a LinearMapAA conforms to the AbstractMatrix
requirement that a setindex! method exists.
Use of this function is strongly discouraged, other than for testing.

2018-01-19, Jeff Fessler, University of Michigan
=#

using Test: @test
using LinearMaps #: LinearMaps.FunctionMap

Indexer = AbstractVector{Int}

"""
`A[i,j] = X`

Mathematically, if B = copy(A) and then we say `B[i,j] = s`, for scalar `s`,
then `B = A + (s - A[i,j]) e_i e_j'`
where `e_i` and `e_j` are the appropriate unit vectors.
Using this basic math, we can perform `B*x` using `A*x` and `A[i,j]`.

This idea generalizes to `B[ii,jj] = X` where `ii` and `jj` are vectors.

This method works because LinearMapAA is a *mutable* struct.
"""
function Base.setindex!(A::LinearMapAA, X::AbstractMatrix,
		ii::Indexer, jj::Indexer)
	# todo: handle WrappedMap differently
	L = A._lmap
	Aij = A[ii,jj] # must extract this outside of forw() to avoid recursion!
	Aij = reshape(Aij, size(X)) # needed for [i,:] case
	if true
		forw = (x) ->
			begin
				tmp = L * x 
				tmp[ii] += (X - Aij) * x[jj]
				return tmp
			end
	end

	has_adj = !(typeof(L) <: LinearMaps.FunctionMap) || (L.fc !== nothing)
	if has_adj
		back = (y) ->
			begin
				tmp = L' * y 
				tmp[jj] += (X - Aij)' * y[ii]
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
		A._lmap = LinearMap(forw, back, size(L)...)
	else
		A._lmap = LinearMap(forw, size(L)...)
	end

	nothing
end


# [i,j] = s
Base.setindex!(A::LinearMapAA, s::Number, i::Int, j::Int) =
	setindex!(A, fill(s,1,1), [i], [j])

# [ii,jj] = X
Base.setindex!(A::LinearMapAA, X::AbstractMatrix,
	ii::AbstractVector{Bool}, jj::AbstractVector{Bool}) =
	setindex!(A, X, findall(ii), findall(jj))

# [:,jj] = X
Base.setindex!(A::LinearMapAA, X::AbstractMatrix, ::Colon, jj::AbstractVector{Bool}) =
	setindex!(A, X, :, findall(jj))
Base.setindex!(A::LinearMapAA, X::AbstractMatrix, ::Colon, jj::Indexer) =
	setindex!(A, X, 1:size(A,1), jj)

# [ii,:] = X
Base.setindex!(A::LinearMapAA, X::AbstractMatrix, ii::AbstractVector{Bool}, ::Colon) =
	setindex!(A, X, findall(ii), :)
Base.setindex!(A::LinearMapAA, X::AbstractMatrix, ii::Indexer, ::Colon) =
	setindex!(A, X, ii, 1:size(A,2))

# [:,j] = v
Base.setindex!(A::LinearMapAA, v::AbstractVector, ::Colon, j::Int) =
	setindex!(A, reshape(v,:,1), :, [j])

# [ii,j] = v
Base.setindex!(A::LinearMapAA, v::AbstractVector, ii::Indexer, j::Int) =
	setindex!(A, reshape(v,:,1), ii, [j])

# [i,:] = v
Base.setindex!(A::LinearMapAA, v::AbstractVector, i::Int, ::Colon) =
	setindex!(A, reshape(v,1,:), [i], :)

# [i,jj] = v
Base.setindex!(A::LinearMapAA, v::AbstractVector, i::Int, jj::Indexer) =
	setindex!(A, reshape(v,1,:), [i], jj)

# [ii,jj] = s
Base.setindex!(A::LinearMapAA, s::Number, ii::AbstractVector{Bool}, jj::Indexer) =
	setindex!(A, s, findall(ii), jj)
Base.setindex!(A::LinearMapAA, s::Number, ii::Indexer, jj::AbstractVector{Bool}) =
	setindex!(A, s, ii, findall(jj))
Base.setindex!(A::LinearMapAA, s::Number,
	ii::AbstractVector{Bool}, jj::AbstractVector{Bool}) =
	setindex!(A, s, findall(ii), findall(jj))
Base.setindex!(A::LinearMapAA, s::Number, ii::Indexer, jj::Indexer) =
	setindex!(A, fill(s,length(ii),length(jj)), ii, jj)

# [:,j] = s
Base.setindex!(A::LinearMapAA, s::Number, ::Colon, j::Int) =
	setindex!(A, s, 1:size(A,1), [j])

# [ii,:] = s
Base.setindex!(A::LinearMapAA, s::Number, ii::AbstractVector{Bool}, ::Colon) =
	setindex!(A, s, findall(ii), :)
Base.setindex!(A::LinearMapAA, s::Number, ii::Indexer, ::Colon) =
	setindex!(A, s, ii, 1:size(A,2))

# [ii,j] = s
Base.setindex!(A::LinearMapAA, s::Number, ii::AbstractVector{Bool}, j::Int) =
	setindex!(A, s, findall(ii), [j])
Base.setindex!(A::LinearMapAA, s::Number, ii::Indexer, j::Int) =
	setindex!(A, fill(s, length(ii), 1), ii, [j])

# [i,:] = s
Base.setindex!(A::LinearMapAA, s::Number, i::Int, ::Colon) =
	setindex!(A, fill(s, 1, size(A,2)), [i], :)

# [i,jj] = s
Base.setindex!(A::LinearMapAA, s::Number, i::Int, jj::AbstractVector{Bool}) =
	setindex!(A, s, i, findall(jj))
Base.setindex!(A::LinearMapAA, s::Number, i::Int, jj::Indexer) =
	setindex!(A, fill(s, 1, length(jj)), [i], jj)

# [:,jj] = s
Base.setindex!(A::LinearMapAA, s::Number, ::Colon, jj::AbstractVector{Bool}) =
	setindex!(A, s, :, findall(jj))
Base.setindex!(A::LinearMapAA, s::Number, ::Colon, jj::Indexer) =
	setindex!(A, s, 1:size(A,1), jj)

# [kk]
#= too much work, so unsupported
Base.setindex!(A::LinearMapAA, v::AbstractVector,
		kk::Indexer) =
=#

# [:] = v
Base.setindex!(A::LinearMapAA, v::AbstractVector, ::Colon) =
	setindex!(A, reshape(v,size(A)), :, :)

# [:] = s
Base.setindex!(A::LinearMapAA, s::Number, ::Colon) =
	begin
		(M,N) = size(A)
		forw = x -> fill(s, M) * (ones(1,N) * x)[1]
		back = y -> fill(conj(s), N) * (ones(1, M) * y)[1]
		A._lmap = LinearMap(forw, back, M, N)
	end

# [:,:] = s
Base.setindex!(A::LinearMapAA, s::Number, ::Colon, ::Colon) =
	setindex!(A, s, :)

# [:,:] = X
Base.setindex!(A::LinearMapAA, X::AbstractMatrix, ::Colon, ::Colon) =
	begin
		A._lmap = LinearMap(X)
	end

# [k] = s
"""
`setindex!(A::LinearMapAA, s::Number, k::Int)`

Provide the single index version to meet the `AbstractArray` spec:
https://docs.julialang.org/en/latest/manual/interfaces/#Indexing-1
"""
Base.setindex!(A::LinearMapAA, s::Number, k::Int) =
	setindex!(A, s, Tuple(CartesianIndices(size(A))[k])...) # any better way?


#=
"""
`setindex!(A::LinearMapAA, v::Number, k::Int)`

Provide the single index version to meet the `AbstractArray` spec:
https://docs.julialang.org/en/latest/manual/interfaces/#Indexing-1
"""
function Base.setindex!(A::LinearMapAA, v::Number, k::Int)
	c = CartesianIndices(size(A))[k] # is there a more elegant way?
	setindex!(A, v::Number, c[1], c[2])
end
=#


"""
`LinearMapAA_test_setindex(A::LinearMapAA)`
"""
function LinearMapAA_test_setindex(A::LinearMapAA)

    @test all(size(A) .>= (4,4)) # required by tests

	tf1 = [false; trues(size(A,1)-1)]
	tf2 = [false; trues(size(A,2)-2); false]
	ii1 = (3, 2:4, [2,4], :, tf1)
	ii2 = (2, 3:4, [1,4], :, tf2)

	# test all [?,?] combinations with array "X"
	for i2 in ii2
		for i1 in ii1
			B = copy(A)
			X = 2 .+ A[i1,i2].^2 # values must differ from A[ii,jj]
			B[i1,i2] = X
			Am = Matrix(A)
			Bm = Matrix(B)
			Am[i1,i2] = X
			@test isapprox(Am, Bm)
		end
	end

	# test all [?,?] combinations with scalar "s"
	for i2 in ii2
		for i1 in ii1
			B = copy(A)
			s = maximum(abs.(A[:])) + 2
			B[i1,i2] = s
			Am = Matrix(A)
			Bm = Matrix(B)
			if (i1 == Colon() || ndims(i1) > 0) || (i2 == Colon() || ndims(i2) > 0)
				Am[i1,i2] .= s
			else
				Am[i1,i2] = s
			end
			@test isapprox(Am, Bm)
		end
	end

	# others not supported for now
	set1 = (3, ) # [3], [2,4], 2:4, (1:length(A)) .== 2), end # todo
	for s1 in set1
		B = copy(A)
		X = 2 .+ A[s1].^2 # values must differ from A[ii,jj]
		B[s1] = X
		Am = Matrix(A)
		Bm = Matrix(B)
		Am[s1] = X
		@test isapprox(Am, Bm)
	end

	# insanity below here

	# A[:] = s
	B = copy(A)
	B[:] = 5
	Am = Matrix(A)
	Am[:] .= 5
	Bm = Matrix(B)
	@test Bm == Am

	# A[:] = v
	B = copy(A)
	v = 1:length(A)
	B[:] = v
	Am = Matrix(A)
	Am[:] .= v
	Bm = Matrix(B)
	@test Bm == Am

	# A[:,:]
	B = copy(A)
	B[:,:] = 6
	Am = Matrix(A)
	Am[:,:] .= 6
	Bm = Matrix(B)
	@test Bm == Am

	true
end
