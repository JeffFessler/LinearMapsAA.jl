#=
lm-aa
2019-08-07 Jeff Fessler, University of Michigan
=#

using LinearMaps: LinearMap
using LinearAlgebra: UniformScaling
import LinearAlgebra: issymmetric, ishermitian, isposdef
import LinearAlgebra: mul!, lmul!, rmul!
import SparseArrays: sparse

"""
`struct LinearMapAA{T, M <: LinearMap, P <: NamedTuple} <: AbstractMatrix{T}`
"""
struct LinearMapAA{T, M <: LinearMap, P <: NamedTuple} <: AbstractMatrix{T}
	_lmap::M
	_prop::P
	function LinearMapAA{T}(L::M, p::P) where {T, M <: LinearMap, P <: NamedTuple}
		new{T,M,P}(L, p)
	end
end


# constructors

"""
`A = LinearMapAA{L::LinearMap, prop::NamedTuple)`
constructor
"""
function LinearMapAA(L::LinearMap, prop::NamedTuple)
	T = eltype(L)
	return LinearMapAA{T}(L, prop)
end

"""
`A = LinearMapAA{L::LinearMap)`
constructor
"""
LinearMapAA(L::LinearMap) = LinearMapAA(L, (none=nothing,))

"""
`A = LinearMapAA(f::Function, fc::Function, M::Int, N::Int, prop::NamedTuple)`
constructor
"""
LinearMapAA(f::Function, fc::Function, M::Int, N::Int, prop::NamedTuple) =
	LinearMapAA(LinearMap(f, fc, M, N), prop)

"""
`A = LinearMapAA(f::Function, fc::Function, M::Int, N::Int)`
constructor
"""
LinearMapAA(f::Function, fc::Function, M::Int, N::Int) =
	LinearMapAA(f, fc, M, N, (none=nothing,))

"""
`A = LinearMapAA(L::AbstractMatrix, prop::NamedTuple)`
constructor
"""
LinearMapAA(L::AbstractMatrix, prop::NamedTuple) =
	LinearMapAA(LinearMap(L), prop)

"""
`A = LinearMapAA(L::AbstractMatrix)`
constructor
"""
LinearMapAA(L::AbstractMatrix) =
	LinearMapAA(LinearMap(L), (none=nothing,))


# size
Base.size(A::LinearMapAA) = size(A._lmap)

# adjoint
Base.adjoint(A::LinearMapAA) = LinearMapAA(A._lmap', A._prop)

# transpose
Base.transpose(A::LinearMapAA) = LinearMapAA(transpose(A._lmap), A._prop)

# eltype
Base.eltype(A::LinearMapAA) = eltype(A._lmap)

# LinearMap algebraic properties
issymmetric(A::LinearMapAA) = issymmetric(A._lmap)
ishermitian(A::LinearMapAA{<:Real}) = issymmetric(A._lmap)
ishermitian(A::LinearMapAA) = ishermitian(A._lmap)
isposdef(A::LinearMapAA) = isposdef(A._lmap)

# comparison of LinearMapAA objects, sufficient but not necessary
Base.:(==)(A::LinearMapAA, B::LinearMapAA) =
	(eltype(A) == eltype(B) && A._lmap == B._lmap && A._prop == B._prop)

# convert to sparse
sparse(A::LinearMapAA) = sparse(A._lmap)

# cat
# todo: requires updated LinearMap

function Base.hcat(As::Union{LinearMapAA,UniformScaling}...)
	tmp = hcat([A._lmap for A in As]...)
	LinearMapAA(hcat([A._lmap for A in As]...), (hcat=nothing,))
end
function Base.vcat(As::Union{LinearMapAA,UniformScaling}...)
	LinearMapAA(vcat([A._lmap for A in As]...), (vcat=nothing,))
end
function Base.hvcat(rows::NTuple{nr,Int}, As::Union{LinearMapAA,UniformScaling}...) where nr
	LinearMapAA(rows, hvcat([A._lmap for A in As]...), (hvcat=nothing,))
end


# multiply with vectors

mul!(y::AbstractVector, A::LinearMapAA, x::AbstractVector) = mul!(y, A._lmap, x)
lmul!(s::Number, A::LinearMapAA) = lmul!(s, A._lmap)
rmul!(A::LinearMapAA, s::Number) = rmul!(A._lmap, s)

#=
function A_mul_B!(y::AbstractVector, A::LinearMapAA, x::AbstractVector)
	A_mul_B!(y, A._lmap, x)
	return y
end

function At_mul_B!(x::AbstractVector, A::LinearMapAA, y::AbstractVector)
	At_mul_B!(x, A._lmap, y)
	return x
end

function Ac_mul_B!(x::AbstractVector, A::LinearMapAA, y::AbstractVector)
	Ac_mul_B!(x, A._lmap, y)
	return x
end
=#


# multiply objects

Base.:(*)(A::LinearMapAA, B::LinearMapAA) = LinearMapAA(A._lmap * B._lmap, (prod=nothing,))
Base.:(*)(A::LinearMapAA, B::AbstractMatrix) = LinearMapAA(A._lmap * LinearMap(B), A._prop)
Base.:(*)(A::AbstractMatrix, B::LinearMapAA) = LinearMapAA(LinearMap(A) * B._lmap, B._prop)


# A.?
Base.getproperty(A::LinearMapAA, s::Symbol) =
	s in (:_lmap, :_prop) ? getfield(A, s) :
#	s == :m ? size(A._lmap, 1) :
	haskey(A._prop, s) ? getfield(A._prop, s) :
		throw("unknown key $s")

Base.propertynames(A::LinearMapAA) = fieldnames(A._prop)


# indexing

# A[end]
function Base.lastindex(A::LinearMapAA)
	return prod(size(A._lmap))
end

# A[?,end] and A[end,?]
function Base.lastindex(A::LinearMapAA, d::Integer)
	return size(A._lmap, d)
end

# A[i,j]
function Base.getindex(A::LinearMapAA, i::Integer, j::Integer)
	T = eltype(A)
	e = zeros(T, size(A._lmap,2)); e[j] = one(T)
	tmp = A._lmap * e
	return tmp[i]
end

# A[:,j]
# it is crucial to provide this function rather than to inherit from
# Base.getindex(A::AbstractArray, ::Colon, ::Integer)
# because Base.getindex does this by iterating (I think).
function Base.getindex(A::LinearMapAA, ::Colon, j::Integer)
	T = eltype(A)
	e = zeros(T, size(A,2)); e[j] = one(T)
	return A * e
end

# A[i,:]
function Base.getindex(A::LinearMapAA, i::Integer, ::Colon)
	# in Julia: A[i,:] = A'[:,i] for real matrix A else need conjugate
	return eltype(A) <: Complex ? conj.(A'[:,i]) : transpose(A)[:,i]
end

# A[:,j:k]
# this one is also important for efficiency
function Base.getindex(A::LinearMapAA, ::Colon, ur::UnitRange)
	return hcat([A[:,j] for j in ur]...)
end

# A[i:k,:]
Base.getindex(A::LinearMapAA, ur::UnitRange, ::Colon) = A'[:,ur]'

# A[:,:] = Matrix(A)
Base.getindex(A::LinearMapAA, ::Colon, ::Colon) = Matrix(A._lmap)


# todo: setindex!


# test
using Test: @test, @test_throws


"""
`LinearMapAA_test_getindex(A::LinearMapAA)`
tests for `getindex`
"""
function LinearMapAA_test_getindex(A::LinearMapAA)
	B = Matrix(A)

	@test all(size(A) .>= (4,4)) # required by tests
	@test B[1] == A[1]
	@test B[7] == A[7]
	@test B[:,5] == A[:,5]
	@test B[3,:] == A[3,:]
	@test B[1,3] == A[1,3]
	@test B[:,1:3] == A[:,1:3]
	@test B[1:3,:] == A[1:3,:]
	@test B[1:3,2:4] == A[1:3,2:4]
	@test B == A[:,:]
	@test B'[3] == A'[3]
	@test B'[:,4] == A'[:,4]
	@test B'[2,:] == A'[2,:]
	@test B[end] == A[end] # lastindex
	@test B[3,end-1] == A[3,end-1]

	# The following rely on the fact that LinearMapAA <:i AbstractMatrix.
	# These indexing modes inherit from general Base.getindex abilities.
	@test B[[1, 3, 4]] == A[[1, 3, 4]]
	@test B[:, [1, 3, 4]] == A[:, [1, 3, 4]]
	@test B[[1, 3, 4], :] == A[[1, 3, 4], :]
	@test B[4:7] == A[4:7]

	true
end


"""
`LinearMapAA_test_vmul(A::LinearMapAA)`
tests for multiply with vector and `lmul!` and `rmul!` for scalars too
"""
function LinearMapAA_test_vmul(A::LinearMapAA)
	B = Matrix(A)

	u = rand(size(A,1))
	v = rand(size(A,2))

	y = A * v
	x = A' * u
	@test isapprox(B * v, y)
	@test isapprox(B' * u, x)

	mul!(y, A, v)
	mul!(x, A', u)
	@test isapprox(B * v, y)
	@test isapprox(B' * u, x)

	s = 5.1
	C = copy(A)
	lmul!(s, C)
	@test isapprox(s * B * v, C * v)

	C = copy(A)
	rmul!(C, s)
	@test isapprox(s * B * v, C * v)

	true
end


"""
`LinearMapAA(:test)`
self test
"""
function LinearMapAA(test::Symbol)
	test != :test && throw(ArgumentError("test $test"))

	B = 1:6
	A = LinearMap(x -> B*x, y -> B'*y, 6, 1)

	N = 6; M = N+1
	forw = x -> [cumsum(x); 0] # non-square to stress test
	back = y -> reverse(cumsum(reverse(y[1:N])))
	A = LinearMap(forw, back, M, N)

	prop = (name="cumsum", extra=1)
	C = LinearMapAA(A, prop)

	@test C == LinearMapAA(forw, back, M, N, prop)
	@test LinearMapAA(forw, back, M, N) isa LinearMapAA

	@test issymmetric(C) == false
	@test ishermitian(C) == false
	@test isposdef(C) == false
	@test issymmetric(C' * C) == true

	Am = Matrix(A)
	@test Matrix(LinearMapAA(Am, prop)) == Am
	@test Matrix(LinearMapAA(Am)) == Am
	@test Matrix(sparse(C)) == Am

	@test eltype(C) == eltype(A)
	@test ndims(C) == 2
	@test size(C) == size(A)

	@test C._prop == prop
	@test C.name == prop.name

	@test_throws String C.bug

	@test Matrix(C)' == Matrix(C')
	@test LinearMapAA_test_getindex(C)
	@test LinearMapAA_test_vmul(C)

	# C2 = [C C]

	D = C * C'
	@test Matrix(D) == Am * Am'
	@test issymmetric(D) == true
	E = C * Am'
	@test Matrix(E) == Am * Am'
	F = Am' * C
	@test Matrix(F) == Am' * Am
	@test LinearMapAA_test_getindex(F)

	true
end
