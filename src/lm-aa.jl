#=
lm-aa
2019-08-07 Jeff Fessler, University of Michigan
=#

export lmaa_hcat, lmaa_vcat, lmaa_hvcat

using LinearMaps: LinearMap
using LinearAlgebra: UniformScaling, I
import LinearAlgebra: issymmetric, ishermitian, isposdef
import LinearAlgebra: mul!, lmul!, rmul!
import SparseArrays: sparse


Indexer = AbstractVector{Int}

"""
`mutable struct LinearMapAA{T} <: AbstractMatrix{T}`

old way may not properly allow `setindex!` to work as desired
because it may change the type of the lmap and of the prop:
`struct LinearMapAA{T, M <: LinearMap, P <: NamedTuple} <: AbstractMatrix{T}`
"""
mutable struct LinearMapAA{T} <: AbstractMatrix{T}
#{T, M <: LinearMap, P <: NamedTuple}
#	_lmap::M
	_lmap::LinearMap
#	_prop::P
	_prop::NamedTuple
#=
	function LinearMapAA{T}(L::M, p::P) where {T, M <: LinearMap, P <: NamedTuple}
	function LinearMapAA(L::LinearMap, p::NamedTuple) # where {T, M <: LinearMap, P <: NamedTuple}
	#	new{T,M,P}(L, p)
		new(L, p)
	end
=#
end

include("setindex.jl")


# constructors

"""
`A = LinearMapAA(L::LinearMap [, prop::NamedTuple ; T = eltype(L)])`
constructor

`prop` cannot include the field `_lmap` or `_prop`
"""
function LinearMapAA(L::LinearMap, prop::NamedTuple ; T = eltype(L))
	:_lmap in propertynames(prop) && throw("cannot use _lmap")
	:_prop in propertynames(prop) && throw("cannot use _prop")
	return LinearMapAA{T}(L, prop)
end
LinearMapAA(L::LinearMap) = LinearMapAA(L, (none=nothing,))

"""
`A = LinearMapAA(L::AbstractMatrix [, prop::NamedTuple])`
constructor
"""
LinearMapAA(L::AbstractMatrix, prop::NamedTuple) =
	LinearMapAA(LinearMap(L), prop)
LinearMapAA(L::AbstractMatrix) = LinearMapAA(L, (none=nothing,))

"""
`A = LinearMapAA(f::Function, fc::Function, D::Dims{2} [, prop::NamedTuple)] ; T::DataType`
constructor
"""
LinearMapAA(f::Function, fc::Function, D::Dims{2}, prop::NamedTuple ;
	T::DataType = Float32) =
	LinearMapAA(LinearMap{T}(f, fc, D[1], D[2]), prop)
LinearMapAA(f::Function, fc::Function, D::Dims{2}, T::DataType = Float32) =
	LinearMapAA(f, fc, D, (none=nothing,) ; T=T)

"""
`A = LinearMapAA(f::Function, D::Dims{2} [, prop::NamedTuple)]`
constructor
"""
LinearMapAA(f::Function, D::Dims{2}, prop::NamedTuple ; T::DataType = Float32) =
	LinearMapAA(LinearMap{T}(f, D[1], D[2]), prop)
LinearMapAA(f::Function, D::Dims{2} ; T::DataType = Float32) =
	LinearMapAA(f, D, (none=nothing,), T=T)


# copy
Base.copy(A::LinearMapAA) = LinearMapAA(A._lmap, A._prop)

# Matrix
Base.Matrix(A::LinearMapAA) = Matrix(A._lmap)

# ndims
# Base.ndims(A::LinearMapAA) = ndims(A._lmap) # 2 for AbstractMatrix

# display
Base.display(A::LinearMapAA) =
	begin
		println("LinearMapAA: $(size(A,1))×$(size(A,2)) $(A._prop)")
		tmp = "$(A._lmap)"[1:77]
		println(" $tmp ..")
	#	display(A._lmap)
	#	display(A._prop)
	end

# size
Base.size(A::LinearMapAA) = size(A._lmap)
Base.size(A::LinearMapAA, d::Int) = size(A._lmap, d)

# adjoint
Base.adjoint(A::LinearMapAA) = LinearMapAA(A._lmap', A._prop)

# transpose
Base.transpose(A::LinearMapAA) = LinearMapAA(transpose(A._lmap), A._prop)

# eltype
Base.eltype(A::LinearMapAA) = eltype(A._lmap)

# LinearMap algebraic properties
issymmetric(A::LinearMapAA) = issymmetric(A._lmap)
#ishermitian(A::LinearMapAA{<:Real}) = issymmetric(A._lmap)
ishermitian(A::LinearMapAA) = ishermitian(A._lmap)
isposdef(A::LinearMapAA) = isposdef(A._lmap)

# comparison of LinearMapAA objects, sufficient but not necessary
Base.:(==)(A::LinearMapAA, B::LinearMapAA) =
	(eltype(A) == eltype(B) && A._lmap == B._lmap && A._prop == B._prop)

# convert to sparse
sparse(A::LinearMapAA) = sparse(A._lmap)

# cat (hcat, vcat, hvcat) are tricky for avoiding type piracy
# It is especially hard to handle AbstractMatrix,
# so typically force the user to wrap it in LinearMap(AA) first.
#LMcat{T} = Union{LinearMapAA{T}, LinearMap{T}, UniformScaling{T},AbstractMatrix{T}}
LMcat{T} = Union{LinearMapAA{T}, LinearMap{T}, UniformScaling{T}} # settle
#LMelse{T} = Union{LinearMap{T},UniformScaling{T},AbstractMatrix{T}} # non AA

# convert to something suitable for LinearMap.*cat
function lm_promote(A::LMcat)
	A isa LinearMapAA ? A._lmap :
	A isa AbstractMatrix ? LinearMap(A) :
	A isa UniformScaling ? A : # leave unchanged - ok for LinearMaps.*cat
	# A isa LinearMap ?
	A # otherwise it is this
end

# single-letter codes for cat objects, e.g., [A I A] becomes "AIA"
#= not so useful
function lm_code(A)
	isa(A, LinearMapAA) ? "A" :
	isa(A, AbstractMatrix) ? "M" :
	isa(A, UniformScaling) ? "I" :
	isa(A, LinearMap) ? "L" :
	"?"
end
=#

# concatenate the single-letter codes, e.g., [A I A] becomes "AIA"
# lm_name = As -> *(lm_code.(As)...)
lm_name = As -> nothing

# these rely on LinearMap.*cat methods
"`B = lmaa_hcat(A1, A2, ...)` `hcat` of multiple objects"
lmaa_hcat(As::LMcat...) =
	LinearMapAA(hcat(lm_promote.(As)...), (hcat=lm_name(As),))
"`B = lmaa_vcat(A1, A2, ...)` `vcat` of multiple objects"
lmaa_vcat(As::LMcat...) =
	LinearMapAA(vcat(lm_promote.(As)...), (vcat=lm_name(As),))
"`B = lmaa_hvcat(rows, A1, A2, ...)` `hvcat` of multiple objects"
lmaa_hvcat(rows::NTuple{nr,Int} where {nr}, As::LMcat...) =
	LinearMapAA(hvcat(rows, lm_promote.(As)...), (hvcat=lm_name(As),))

# a single leading LinearMapAA followed by others is clear
Base.hcat(A1::LinearMapAA, As::LMcat...) =
	lmaa_hcat(A1, As...)
Base.vcat(A1::LinearMapAA, As::LMcat...) =
	lmaa_vcat(A1, As...)
Base.hvcat(rows::NTuple{nr,Int} where {nr}, A1::LinearMapAA, As::LMcat...) =
	lmaa_hvcat(rows, A1, As...)
# or in 2nd position, beyond that, user can use lmaa_*
#Base.hcat(A1::LMelse, A2::LinearMapAA, As::LMcat...) = # fails!?
Base.hcat(A1::UniformScaling, A2::LinearMapAA, As::LMcat...) =
	lmaa_hcat(A1, A2, As...)
Base.vcat(A1::UniformScaling, A2::LinearMapAA, As::LMcat...) =
	lmaa_vcat(A1, A2, As...)
Base.hvcat(rows::NTuple{nr,Int} where nr,
		A1::UniformScaling, A2::LinearMapAA, As::LMcat...) =
	lmaa_hvcat(rows, A1, A2, As...)

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


# add or subtract objects
Base.:(+)(A::LinearMapAA, B::LinearMapAA) =
	LinearMapAA(A._lmap + B._lmap, (sum=nothing,))
Base.:(+)(A::LinearMapAA, B::AbstractMatrix) =
	LinearMapAA(A._lmap + LinearMap(B), A._prop)
Base.:(+)(A::AbstractMatrix, B::LinearMapAA) =
	LinearMapAA(LinearMap(A) + B._lmap, B._prop)
Base.:(-)(A::LinearMapAA, B::LinearMapAA) = A + (-1)*B
Base.:(-)(A::LinearMapAA, B::AbstractMatrix) = A + (-1)*B
Base.:(-)(A::AbstractMatrix, B::LinearMapAA) = A + (-1)*B

# multiply objects
Base.:(*)(A::LinearMapAA, B::LinearMapAA) =
	LinearMapAA(A._lmap * B._lmap, (prod=nothing,))
Base.:(*)(A::LinearMapAA, B::AbstractMatrix) =
	LinearMapAA(A._lmap * LinearMap(B), A._prop)
Base.:(*)(A::AbstractMatrix, B::LinearMapAA) =
	LinearMapAA(LinearMap(A) * B._lmap, B._prop)

# multiply with I or s*I
Base.:(*)(A::LinearMapAA, B::UniformScaling) = LinearMapAA(A._lmap * B, A._prop)
Base.:(*)(B::UniformScaling, A::LinearMapAA) = LinearMapAA(B * A._lmap, A._prop)

# multiply with vector
Base.:(*)(A::LinearMapAA, v::AbstractVector{<:Number}) = A._lmap * v

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

# [end]
function Base.lastindex(A::LinearMapAA)
	return prod(size(A._lmap))
end

# [?,end] and [end,?]
function Base.lastindex(A::LinearMapAA, d::Int)
	return size(A._lmap, d)
end

# A[i,j]
function Base.getindex(A::LinearMapAA, i::Int, j::Int)
	T = eltype(A)
	e = zeros(T, size(A._lmap,2)); e[j] = one(T)
	tmp = A._lmap * e
	return tmp[i]
end

# A[:,j]
# it is crucial to provide this function rather than to inherit from
# Base.getindex(A::AbstractArray, ::Colon, ::Int)
# because Base.getindex does this by iterating (I think).
function Base.getindex(A::LinearMapAA, ::Colon, j::Int)
	T = eltype(A)
	e = zeros(T, size(A,2)); e[j] = one(T)
	return A * e
end

# A[ii,j]
Base.getindex(A::LinearMapAA, ii::Indexer, j::Int) = A[:,j][ii]

# A[i,jj]
Base.getindex(A::LinearMapAA, i::Int, jj::Indexer) = A[i,:][jj]

# A[:,jj]
# this one is also important for efficiency
Base.getindex(A::LinearMapAA, ::Colon, jj::AbstractVector{Bool}) =
	A[:,findall(jj)]
Base.getindex(A::LinearMapAA, ::Colon, jj::Indexer) =
	hcat([A[:,j] for j in jj]...)

# A[ii,:]
# trick: cannot use A' for a FunctionMap with no fc
function Base.getindex(A::LinearMapAA, ii::Indexer, ::Colon)
	if (:fc in propertynames(A._lmap)) && isnothing(A._lmap.fc)
		return hcat([A[ii,j] for j in 1:size(A,2)]...)
	else
		return A'[:,ii]'
	end
end

# A[ii,jj]
Base.getindex(A::LinearMapAA, ii::Indexer, jj::Indexer) = A[:,jj][ii,:]

# A[k]
function Base.getindex(A::LinearMapAA, k::Int)
	c = CartesianIndices(size(A._lmap))[k] # is there a more elegant way?
	return A[c[1], c[2]]
end

# A[kk]
Base.getindex(A::LinearMapAA, kk::AbstractVector{Bool}) = A[findall(kk)]
Base.getindex(A::LinearMapAA, kk::Indexer) = [A[k] for k in kk]

# A[i,:]
# trick: one row slice returns a 1D ("column") vector
Base.getindex(A::LinearMapAA, i::Int, ::Colon) = A[[i],:][:]

# A[:,:] = Matrix(A)
Base.getindex(A::LinearMapAA, ::Colon, ::Colon) = Matrix(A._lmap)

# A[:]
Base.getindex(A::LinearMapAA, ::Colon) = A[:,:][:]


# test
using Test: @test, @test_throws


"""
`LinearMapAA_test_getindex(A::LinearMapAA)`
tests for `getindex`
"""
function LinearMapAA_test_getindex(A::LinearMapAA)
	B = Matrix(A)
	@test all(size(A) .>= (4,4)) # required by tests

	tf1 = [false; trues(size(A,1)-1)]
	tf2 = [false; trues(size(A,2)-2); false]
	ii1 = (3, 2:4, [2,4], :, tf1)
	ii2 = (2, 3:4, [1,4], :, tf2)
	for i2 in ii2
		for i1 in ii1
		#	@show i1,i2
			@test B[i1,i2] == A[i1,i2]
		end
	end

	L = A._lmap
	test_adj = !((:fc in propertynames(L)) && isnothing(L.fc))
	if test_adj
		for i1 in ii2
			for i2 in ii1
				@test B'[i1,i2] == A'[i1,i2]
			end
		end
	end

	# "end"
	@test B[3,end-1] == A[3,end-1]
	@test B[end-2,3] == A[end-2,3]
	if test_adj
		@test B'[3,end-1] == A'[3,end-1]
	end

	# [?]
	@test B[1] == A[1]
	@test B[7] == A[7]
	if test_adj
		@test B'[3] == A'[3]
	end
	@test B[end] == A[end] # lastindex

	kk = [k in [3,5] for k = 1:length(A)] # Bool
	@test B[kk] == A[kk]

	# Some tests could rely on the fact that LinearMapAA <:i AbstractMatrix,
	# by inheriting from general Base.getindex, but all are provided here.
	@test B[[1, 3, 4]] == A[[1, 3, 4]]
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

#=
	this approach using eval() works only in the global scope!

	N = 6
	forw = x -> [cumsum(x); 0] # non-square to stress test
	back = y -> reverse(cumsum(reverse(y[1:N])))
	prop = (name="cumsum", extra=1)
	A = LinearMapAA(forw, back, (N+1, N), prop)

	list1 = [
		:([A I]), :([I A]), :([2*A 3*A]),
		:([A; I]), :([I; A]), :([2*A; 3*A]),
		:([A A I]), :([A I A]), :([2*A 3*A 4*A]),
		:([A; A; I]), :([A; I; A]), :([2*A; 3*A; 4*A]),
		:([I A I]), :([I A A]),
		:([I; A; I]), :([I; A; A]),
	#	:([I I A]), :([I; I; A]), # unsupported
		:([A A; A A]), :([A I; 2A 3I]), :([I A; 2I 3A]),
	#	:([I A; A I]), :([A I; I A]), # weird sizes
		:([A I A; 2A 3I 4A]), :([I A I; 2I 3A 4I]),
	#	:([A I A; I A A]), :([I A 2A; 3A 4I 5A]), # weird
	#	:([I I A; I A I]), # unsupported
		:([A A I; 2A 3A 4I]),
		:([A I I; 2A 3I 4I]),
	]

	M = Matrix(A)
	list2 = [
		:([M I]), :([I M]), :([2*M 3*M]),
		:([M; I]), :([I; M]), :([2*M; 3*M]),
		:([M M I]), :([M I M]), :([2*M 3*M 4*M]),
		:([M; M; I]), :([M; I; M]), :([2*M; 3*M; 4*M]),
		:([I M I]), :([I M M]),
		:([I; M; I]), :([I; M; M]),
	#	:([I I M]), :([I; I; M]), # unsupported
		:([M M; M M]), :([M I; 2M 3I]), :([I M; 2I 3M]),
	#	:([I M; M I]), :([M I; I M]), # weird sizes
		:([M I M; 2M 3I 4M]), :([I M I; 2I 3M 4I]),
	#	:([M I M; I M M]), :([I M 2M; 3M 4I 5M]), # weird
	#	:([I I M; I M I]), # unsupported
		:([M M I; 2M 3M 4I]),
		:([M I I; 2M 3I 4I]),
	]

	length(list1) != length(list2) && throw("bug")

	for ii in 1:length(list1)
		e1 = list1[ii]
		b1 = eval(e1)
		e2 = list2[ii]
		b2 = eval(e2)
		@test b1 isa LinearMapAA
	end
=#


"""
`LinearMapAA_test_cat(A::LinearMapAA)`
test hcat vcat hvcat
"""
function LinearMapAA_test_cat(A::LinearMapAA)
#	L = LinearMap(x -> A*x, y -> A'*y, size(A,1), size(A,2))
	M = Matrix(A)

#=
	# cannot get cat with AbstractMatrix to work
	M1 = reshape(1:35, N+1, N-1)
	H2 = [A M1]
	@test H2 isa LinearMapAA
	@test Matrix(H2) == [Matrix(A) H2]
	H1 = [M1 A]
	@test H1 isa LinearMapAA
	@test Matrix(H1) == [M1 Matrix(A)]

	M2 = reshape(1:(3*N), 3, N)
	V1 = [M2; A]
	@test V1 isa LinearMapAA
	@test Matrix(V1) == [M2; Matrix(A)]
	V2 = [A; M2]
	@test V2 isa LinearMapAA
	@test Matrix(V2) == [Matrix(A); M2]
=#

	fun0 = A -> [
		[A I], [I A], [2A 3A],
		[A; I], [I; A], [2A; 3A],
		[A A I], [A I A], [2A 3A 4A],
		[A; A; I], [A; I; A], [2A; 3A; 4A],
		[I A I], [I A A],
		[I; A; I], [I; A; A],
	#	[I I A], [I; I; A], # unsupported
		[A A; A A], [A I; 2A 3I], [I A; 2I 3A],
	#	[I A; A I], [A I; I A], # weird sizes
		[A I A; 2A 3I 4A], [I A I; 2I 3A 4I],
	#	[A I A; I A A], [I A 2A; 3A 4I 5A], # weird
	#	[I I A; I A I], # unsupported
		[A A I; 2A 3A 4I],
		[A I I; 2A 3I 4I],
	]

	list1 = fun0(A)
	list2 = fun0(M)

	for ii in 1:length(list1)
		b1 = list1[ii]
		b2 = list2[ii]
		@test b1 isa LinearMapAA
		@test b2 isa AbstractMatrix
		@test Matrix(b1) == b2
		@test Matrix(b1') == Matrix(b1)'
	end

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

	B = reshape(1:6, 6, 1)
	@test Matrix(LinearMapAA(B)) == B

	N = 6; M = N+1
	forw = x -> [cumsum(x); 0] # non-square to stress test
	back = y -> reverse(cumsum(reverse(y[1:N])))

	prop = (name="cumsum", extra=1)
	@test LinearMapAA(forw, (M, N)) isa LinearMapAA
	@test LinearMapAA(forw, (M, N), prop, T=Float64) isa LinearMapAA

	L = LinearMap{Float32}(forw, back, M, N)
	A = LinearMapAA(L, prop)
	Lm = Matrix(L)

#	@test lm_name((Lm, A, I, L, "")) == "MAIL?"

	display(A)

	@test A._lmap == LinearMapAA(L)._lmap
#	@test A == LinearMapAA(forw, back, M, N, prop)
	@test A._prop == LinearMapAA(forw, back, (M, N), prop)._prop
	@test A._lmap == LinearMapAA(forw, back, (M, N), prop)._lmap
	@test A == LinearMapAA(forw, back, (M, N), prop)
	@test A._lmap == LinearMapAA(forw, back, (M, N))._lmap
	@test LinearMapAA(forw, back, (M, N)) isa LinearMapAA
	@test propertynames(A) == (:name, :extra)

	@test issymmetric(A) == false
	@test ishermitian(A) == false
	@test ishermitian(im * A) == false
	@test isposdef(A) == false
	@test issymmetric(A' * A) == true

	@test Matrix(LinearMapAA(L, prop)) == Lm
	@test Matrix(LinearMapAA(L)) == Lm
	@test Matrix(sparse(A)) == Lm

	@test eltype(A) == eltype(L)
	@test Base.eltype(A) == eltype(L) # codecov
	@test ndims(A) == 2
	@test size(A) == size(L)

	B = copy(A)
	@test B == A
	@test !(B === A)

	@test A._prop == prop
	@test A.name == prop.name

	@test_throws String A.bug

	@test Matrix(A)' == Matrix(A')
	@test Matrix(A)' == Matrix(transpose(A))
	@test LinearMapAA_test_getindex(A)
	@test LinearMapAA_test_vmul(A)

	@test LinearMapAA_test_cat(A)
	@test LinearMapAA_test_setindex(A)

	# add / subtract
	@test 2A + 6A isa LinearMapAA
	@test 7A - 2A isa LinearMapAA
	@test Matrix(2A + 6A) == 8 * Matrix(A)
	@test Matrix(7A - 2A) == 5 * Matrix(A)
	@test Matrix(7A - 2*ones(size(A))) == 7 * Matrix(A) - 2*ones(size(A))
	@test Matrix(3*ones(size(A)) - 5A) == 3*ones(size(A)) - 5 * Matrix(A)

	# multiply
	@test Matrix(A * 6I) == 6 * Matrix(A)
	@test Matrix(7I * A) == 7 * Matrix(A)
#	@test I * A === A
#	@test A * I === A
	D = A * A'
	@test D isa LinearMapAA
	@test Matrix(D) == Lm * Lm'
	@test issymmetric(D) == true
	E = A * Lm'
	@test E isa LinearMapAA
	@test Matrix(E) == Lm * Lm'
	F = Lm' * A
	@test F isa LinearMapAA
	@test Matrix(F) == Lm' * Lm
	@test LinearMapAA_test_getindex(F)

	# non-adjoint version
	Af = LinearMapAA(forw, (M, N))
	@test LinearMapAA_test_getindex(Af)
	@test LinearMapAA_test_setindex(Af)


#= todo: cut
	a = ones(3,2)
	b = zeros(3,4)

	A = LinearMapAA(x -> a*x, y -> a'*y, (3, 2))
	B = LinearMapAA(x -> b*x, y -> b'*y, (3, 4))

	Al = [a b I]
	C = [A B I] # now returns a new LinearMap :)
	@test typeof(C) <: LinearMapAA
#	display(C)
#	@show C
	@test Matrix(C) == Matrix(Al)
	@test size(C) == (3,2+4+3)
	@test C * ones(size(C,2)) == 3. * ones(3)
	@test C' * ones(size(C,1)) == [3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
	@test Matrix(C)' == Matrix(C')
=#

	true
end

#= mwe for LinearMaps bug
A = LinearMap(ones(3,2))
B = [I A; A I]
Matrix(B) # works fine
Matrix(B') # fails with error DimensionMismatch("mul!")
=#
