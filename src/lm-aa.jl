#=
lm-aa
2019-08-07 Jeff Fessler, University of Michigan
=#

using LinearMaps: LinearMap
using LinearAlgebra: UniformScaling, I
import LinearAlgebra: issymmetric, ishermitian, isposdef
import LinearAlgebra: mul!, lmul!, rmul!
import SparseArrays: sparse


"""
`struct LinearMapAA{T, M <: LinearMap, P <: NamedTuple} <: AbstractMatrix{T}`
"""
mutable struct LinearMapAA{T, M <: LinearMap, P <: NamedTuple} <: AbstractMatrix{T}
#	_lmap::M
	_lmap::LinearMap
	_prop::P
	function LinearMapAA{T}(L::M, p::P) where {T, M <: LinearMap, P <: NamedTuple}
		new{T,M,P}(L, p)
	end
end

include("setindex.jl")


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
`A = LinearMapAA(f::Function, fc::Function, D::Dims{2}, prop::NamedTuple)`
constructor
"""
LinearMapAA(f::Function, fc::Function, D::Dims{2}, prop::NamedTuple) =
	LinearMapAA(LinearMap(f, fc, D[1], D[2]), prop)

"""
`A = LinearMapAA(f::Function, fc::Function, D::Dims{2})`
constructor
"""
LinearMapAA(f::Function, fc::Function, D::Dims{2}) =
	LinearMapAA(f, fc, D[1], D[2], (none=nothing,))

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


# copy
Base.copy(A::LinearMapAA) = LinearMapAA(A._lmap, A._prop)

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

#= these seem pointless; see multiplication with scalars below
lmul!(s::Number, A::LinearMapAA) = lmul!(s, A._lmap)
rmul!(A::LinearMapAA, s::Number) = rmul!(A._lmap, s)
=#

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

# multiply with scalars
Base.:(*)(s::Number, A::LinearMapAA) = LinearMapAA(s*I * A._lmap, A._prop)
Base.:(*)(A::LinearMapAA, s::Number) = LinearMapAA(A._lmap * (s*I), A._prop)


# A.?
Base.getproperty(A::LinearMapAA, s::Symbol) =
	s in (:_lmap, :_prop) ? getfield(A, s) :
#	s == :m ? size(A._lmap, 1) :
	haskey(A._prop, s) ? getfield(A._prop, s) :
		throw("unknown key $s")

Base.propertynames(A::LinearMapAA) = propertynames(A._prop)


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
	C = s * A
	@test isapprox(Matrix(C), s * B)
	C = A * s
	@test isapprox(Matrix(C), B * s)

#=
	s = 5.1
	C = copy(A)
	lmul!(s, C)
	@test isapprox(s * B * v, C * v)

	C = copy(A)
	rmul!(C, s)
	@test isapprox(s * B * v, C * v)
=#

	true
end


"""
`LinearMapAA(:test)`
self test
"""
function LinearMapAA(test::Symbol)
	test != :test && throw(ArgumentError("test $test"))

	B = 1:6
	L = LinearMap(x -> B*x, y -> B'*y, 6, 1)

	N = 6; M = N+1
	forw = x -> [cumsum(x); 0] # non-square to stress test
	back = y -> reverse(cumsum(reverse(y[1:N])))
	L = LinearMap(forw, back, M, N)

	prop = (name="cumsum", extra=1)
	A = LinearMapAA(L, prop)

	@test A._lmap == LinearMapAA(L)._lmap
	@test A == LinearMapAA(forw, back, M, N, prop)
	@test A == LinearMapAA(forw, back, (M, N), prop)
	@test A._lmap == LinearMapAA(forw, back, (M, N))._lmap
	@test LinearMapAA(forw, back, M, N) isa LinearMapAA
	@test propertynames(A) == (:name, :extra)

	@test issymmetric(A) == false
	@test ishermitian(A) == false
	@test ishermitian(im * A) == false
	@test isposdef(A) == false
	@test issymmetric(A' * A) == true

	Lm = Matrix(L)
	@test Matrix(LinearMapAA(Lm, prop)) == Lm
	@test Matrix(LinearMapAA(Lm)) == Lm
	@test Matrix(sparse(A)) == Lm

	@test eltype(A) == eltype(L)
	@test Base.eltype(A) == eltype(L)
	@test ndims(A) == 2
	@test size(A) == size(L)

	@test A._prop == prop
	@test A.name == prop.name

	@test_throws String A.bug

	@test Matrix(A)' == Matrix(A')
	@test LinearMapAA_test_getindex(A)
	@test LinearMapAA_test_vmul(A)

#	@test LinearMapAA_test_setindex(A) # todo

	# todo: cat
	# A2 = [A A]

	D = A * A'
	@test Matrix(D) == Lm * Lm'
	@test issymmetric(D) == true
	E = A * Lm'
	@test Matrix(E) == Lm * Lm'
	F = Lm' * A
	@test Matrix(F) == Lm' * Lm
	@test LinearMapAA_test_getindex(F)

	true
end

#= todo: failing
	N = 4; M = N+1
	forw = x -> [cumsum(x); 0] # non-square to stress test
	back = y -> reverse(cumsum(reverse(y[1:N])))
	L = LinearMap(forw, back, M, N)
	A = LinearMapAA(L, (test=true,))
	B = copy(A)
	B[1,2] = 5
	B
=#
