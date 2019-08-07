#=
lm-aa
2019-08-07 Jeff Fessler, University of Michigan
=#

using LinearMaps: LinearMap
using LinearAlgebra: UniformScaling

"""
`struct LinearMapAA{T, L <: LinearMap, P <: NamedTuple} <: AbstractMatrix{T}`
"""
struct LinearMapAA{T, L <: LinearMap, P <: NamedTuple} <: AbstractMatrix{T}
	lmap::L
	prop::P
	function LinearMapAA{T}(A::L, p::P) where {T, L <: LinearMap, P <: NamedTuple}
		new{T,L,P}(A, p)
	end
end

Base.size(A::LinearMapAA) = size(A.lmap)
Base.adjoint(A::LinearMapAA) = LinearMapAA(A.lmap', A.prop)

# todo: issymmetric etc.


# comparison of LinearMapAA objects, sufficient but not necessary
Base.:(==)(A::LinearMapAA, B::LinearMapAA) = (eltype(A) == eltype(B) && A.lmap == B.lmap && A.prop == B.prop)


# cat
# todo: requires updated LinearMap

function Base.hcat(As::Union{LinearMapAA,UniformScaling}...)
	tmp = hcat([A.lmap for A in As]...)
	LinearMapAA(hcat([A.lmap for A in As]...), (hcat=nothing,))
end
function Base.vcat(As::Union{LinearMapAA,UniformScaling}...)
	LinearMapAA(vcat([A.lmap for A in As]...), (vcat=nothing,))
end
function Base.hvcat(rows::NTuple{nr,Int}, As::Union{LinearMapAA,UniformScaling}...) where nr
	LinearMapAA(rows, hvcat([A.lmap for A in As]...), (hvcat=nothing,))
end


# multiply with vectors

function A_mul_B!(y::AbstractVector, A::LinearMapAA, x::AbstractVector)
	A_mul_B!(y, A.lmap, x)
	return y
end

#function At_mul_B!(y::AbstractVector, A::BlockMap, x::AbstractVector)

function Ac_mul_B!(x::AbstractVector, A::LinearMapAA, y::AbstractVector)
	Ac_mul_B!(x, A.lmap', y)
	return x
end


# multiply objects

Base.:(*)(A::LinearMapAA, B::LinearMapAA) = LinearMapAA(A.lmap * B.lmap, (prod=nothing,))
Base.:(*)(A::LinearMapAA, B::AbstractMatrix) = LinearMapAA(A.lmap * LinearMap(B), A.prop)
Base.:(*)(A::AbstractMatrix, B::LinearMapAA) = LinearMapAA(LinearMap(A) * B.lmap, B.prop)


# A.?
Base.getproperty(A::LinearMapAA, s::Symbol) =
	s in (:lmap, :prop) ? getfield(A, s) :
#	s == :m ? size(A.lmap, 1) :
	haskey(A.prop, s) ? getfield(A.prop, s) :
		throw("unknown key $s")

Base.propertynames(A::LinearMapAA) = fieldnames(A.prop)


# todo: setindex!

# indexing

# A[end]
function Base.lastindex(A::LinearMapAA)
	return prod(size(A.lmap))
end

# A[?,end] and A[end,?]
function Base.lastindex(A::LinearMapAA, d::Integer)
	return size(A.lmap, d)
end

# A[i,j]
function Base.getindex(A::LinearMapAA, i::Integer, j::Integer)
	T = eltype(A.lmap)
	e = zeros(T, size(A.lmap,2)); e[j] = one(T)
	tmp = A.lmap * e
	return tmp[i]
end

# A[:,j]
# it is crucial to provide this function rather than to inherit from
# Base.getindex(A::AbstractArray, ::Colon, ::Integer)
# because Base.getindex does this by iterating (I think).
function Base.getindex(A::LinearMapAA, ::Colon, j::Integer)
	e = zeros(T, size(A,2)); e[j] = one(T)
	return A * e
end

# A[i,:]
function Base.getindex(A::LinearMapAA, i::Integer, ::Colon)
	# in Julia: A[i,:] = A'[:,i] for real matrix A else need conjugate
	return eltype(A.lmap) <: Complex ? conj.(A.lmap'[:,i]) : A.lmap'[:,i]
end

# A[:,j:k]
# this one is also important for efficiency
function Base.getindex(A::LinearMapAA, ::Colon, ur::UnitRange)
	return hcat([A.lmap[:,j] for j in ur]...)
end

# A[i:k,:]
Base.getindex(A::LinearMapAA, ur::UnitRange, ::Colon) = A.lmap'[:,ur]'

# A[:,:] = Matrix(A)
Base.getindex(A::LinearMapAA, ::Colon, ::Colon) = Matrix(A.lmap)


"""
`A = LinearMapAA{L::LinearMap, prop::NamedTuple)`
"""
function LinearMapAA(A::LinearMap, prop::NamedTuple)
	T = eltype(A)
	return LinearMapAA{T}(A, prop)
end

"""
`A = LinearMapAA{L::LinearMap)`
"""
LinearMapAA(A::LinearMap) = LinearMapAA(A, (none=nothing,))


# test
using Test: @test, @test_throws

"""
`LinearMapAA(:test)`
self test
"""
function LinearMapAA(test::Symbol)
	test != :test && throw(ArgumentError("test $test"))

	B = collect(1:6)
	B = reshape(B, 6, 1)
	A = LinearMap(x -> B*x, y -> B'*y, 6, 1)

	N = 6
	A = LinearMap(cumsum, y -> reverse(cumsum(reverse(y))), N)

	prop = (name="cumsum",)
	C = LinearMapAA(A, prop)

	@test ndims(C) == 2
	@test size(C) == size(A)

	@test C.prop == prop
	@test C.name == prop.name

	@test_throws String C.bug

	# C2 = [C C]
	Am = Matrix(A)
	D = C * C'
	@test Matrix(D) == Am * Am'
	E = C * Am'
	@test Matrix(E) == Am * Am'
	F = Am' * C
	@test Matrix(F) == Am' * Am

	true
end
