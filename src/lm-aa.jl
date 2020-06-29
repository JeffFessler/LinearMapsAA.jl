#=
lm-aa
Core methods for LinearMapAA objects.
2019-08-07 Jeff Fessler, University of Michigan
=#

using LinearMaps: LinearMap
using LinearAlgebra: UniformScaling, I
import LinearAlgebra: issymmetric, ishermitian, isposdef
#import LinearAlgebra: mul! #, lmul!, rmul!
import SparseArrays: sparse
using Test: @test, @testset, @test_throws

export redim, undim


# copy
Base.copy(A::LinearMapAX{T,Do,Di}) where {T,Do,Di} =
	LinearMapAA(A._lmap ; prop=A._prop, T=T, idim=A._idim, odim=A._odim)

# Matrix
Base.Matrix(A::LinearMapAX) = Matrix(A._lmap)

# ndims
# Base.ndims(A::LinearMapAX) = ndims(A._lmap) # 2 for AbstractMatrix
Base.ndims(A::LinearMapAO) = ndims(A._lmap) # 2


"""
    show(io::IO, A::LinearMapAX)
    show(io::IO, ::MIME"text/plain", A::LinearMapAX)
Pretty printing for `display`
"""
function Base.show(io::IO, A::LinearMapAX) # short version
	print(io, isa(A, LinearMapAM) ? "LinearMapAM" : "LinearMapAO")
	print(io, ": $(size(A,1)) × $(size(A,2))")
end

# multi-line version:
Base.show(io::IO, ::MIME"text/plain", A::LinearMapAX{T,Do,Di}) where {T,Do,Di} =
	begin
		show(io, A)
		(A._prop != NamedTuple()) && print(io, "\n$(A._prop)")
		print(io, "\nodim=$(A._odim) idim=$(A._idim) T=$T Do=$Do Di=$Di")
	#	print(io, "\n$(A._lmap)\n") # todo: hide until "show" fixed for LM
		print(io, "\n$(typeof(A._lmap)) ...\n")
	#	tmp = "$(A._lmap)"[1:77]
	#	print(io, " $tmp ..")
	end

# size
Base.size(A::LinearMapAX) = size(A._lmap)
Base.size(A::LinearMapAX, d::Int) = size(A._lmap, d)

"""
    redim(A::LinearMapAX ; idim::Dims=A._idim, odim::Dims=A._odim)

"Reinterpret" the `idim` and `odim` of `A`
"""
function redim(A::LinearMapAX{T} ;
	idim::Dims=A._idim, odim::Dims=A._odim) where {T}

	prod(idim) == prod(A._idim) || throw("incompatible idim")
	prod(odim) == prod(A._odim) || throw("incompatible odim")
	return LinearMapAA(A._lmap ; prop=A._prop, T=T, idim=idim, odim=odim)
end

"""
    undim(A::LinearMapAX)

"Reinterpret" the `idim` and `odim` of `A` to be of AM type
"""
undim(A::LinearMapAX{T}) where {T} =
	LinearMapAA(A._lmap ; prop=A._prop, T=T)


# adjoint
Base.adjoint(A::LinearMapAX) =
	LinearMapAA(adjoint(A._lmap), A._prop ; idim=A._odim, odim=A._idim)

# transpose
Base.transpose(A::LinearMapAX) =
	LinearMapAA(transpose(A._lmap), A._prop ; idim=A._odim, odim=A._idim)

# eltype
Base.eltype(A::LinearMapAX) = eltype(A._lmap)

# LinearMap algebraic properties
issymmetric(A::LinearMapAX) = issymmetric(A._lmap)
#ishermitian(A::LinearMapAX{<:Real}) = issymmetric(A._lmap)
ishermitian(A::LinearMapAX) = ishermitian(A._lmap)
isposdef(A::LinearMapAX) = isposdef(A._lmap)

# comparison of LinearMapAX objects, sufficient but not necessary
Base.:(==)(A::LinearMapAX, B::LinearMapAX) =
	eltype(A) == eltype(B) &&
		A._lmap == B._lmap && A._prop == B._prop &&
		A._idim == B._idim && A._odim == B._odim


# convert to sparse
sparse(A::LinearMapAX) = sparse(A._lmap)


# add or subtract objects (with compatible idim,odim)
function Base.:(+)(A::LinearMapAX, B::LinearMapAX)
	(A._idim != B._idim) && throw("idim mismatch in +")
	(A._odim != B._odim) && throw("odim mismatch in +")
	LinearMapAA(A._lmap + B._lmap ;
		idim=A._idim, odim=A._odim,
		prop = (sum=nothing,props=(A._prop,B._prop)),
	)
end

# Allow LMAA + AM only if Do=Di=1
function Base.:(+)(A::LinearMapAX, B::AbstractMatrix)
	(length(A._idim) != 1 || length(A._odim) != 1) && throw("use redim")
	LinearMapAA(A._lmap + LinearMap(B), A._prop)
end

# But allow LMAA + I for any Do,Di
Base.:(+)(A::LinearMapAX, B::UniformScaling) = # A + I -> A + I(N)
	LinearMapAA(A._lmap + B(size(A,2)) ;
		prop = (Aprop=A._prop, Iscale=B(size(A,2))[1]),
		idim = A._idim,
		odim = A._odim,
	)
Base.:(+)(A::AbstractMatrix, B::LinearMapAX) = B + A

Base.:(-)(A::LinearMapAX, B::LinearMapAX) = A + (-1)*B
Base.:(-)(A::LinearMapAX, B::AbstractMatrix) = A + (-1)*B
Base.:(-)(A::AbstractMatrix, B::LinearMapAX) = A + (-1)*B


# A.?
Base.getproperty(A::LinearMapAX, s::Symbol) =
	(s in LMAAkeys) ? getfield(A, s) :
#	s == :m ? size(A._lmap, 1) :
	haskey(A._prop, s) ? getfield(A._prop, s) :
		throw("unknown key $s")

Base.propertynames(A::LinearMapAX) = (propertynames(A._prop)..., LMAAkeys...)



# test


"""
    LinearMapAA(:test)
self test
"""
function LinearMapAA(test::Symbol)
	test != :test && throw(ArgumentError("test $test"))

	B = 1:6
	L = LinearMap(x -> B*x, y -> B'*y, 6, 1)

	B = reshape(1:6, 6, 1)
	@test Matrix(LinearMapAA(B)) == B

	N = 6; M = N+1 # non-square to stress test
	forw = x -> [cumsum(x); 0]
	back = y -> reverse(cumsum(reverse(y[1:N])))

	prop = (name="cumsum", extra=1)
	@test LinearMapAA(forw, (M, N)) isa LinearMapAX
	@test LinearMapAA(forw, (M, N), prop ; T=Float64) isa LinearMapAX

	L = LinearMap{Float32}(forw, back, M, N)
	A = LinearMapAA(L, prop)
	Lm = Matrix(L)

	show(isinteractive() ? stdout : devnull, "text/plain", A)

    @testset "basics" begin
        @test A._lmap == LinearMapAA(L)._lmap
    #   @test A == LinearMapAA(forw, back, M, N, prop)
        @test A._prop == LinearMapAA(forw, back, (M, N), prop)._prop
        @test A._lmap == LinearMapAA(forw, back, (M, N), prop)._lmap
        @test A == LinearMapAA(forw, back, (M, N), prop)
        @test A._lmap == LinearMapAA(forw, back, (M, N))._lmap
        @test LinearMapAA(forw, back, (M, N)) isa LinearMapAX
    end

	@testset "symmetry" begin
		@test issymmetric(A) == false
		@test ishermitian(A) == false
		@test ishermitian(im * A) == false
		@test isposdef(A) == false
		@test issymmetric(A' * A) == true
	end

	@testset "convert" begin
		@test Matrix(LinearMapAA(L, prop)) == Lm
		@test Matrix(LinearMapAA(L)) == Lm
		@test Matrix(sparse(A)) == Lm
	end

	@testset "getproperty" begin
		@test propertynames(A) == (:name, :extra, LMAAkeys...)
		@test A._prop == prop
		@test A.name == prop.name
		@test eltype(A) == eltype(L)
		@test Base.eltype(A) == eltype(L) # codecov
		@test ndims(A) == 2
		@test size(A) == size(L)
    	@test redim(A) isa LinearMapAX
	end

	@testset "copy" begin
		B = copy(A)
		@test B == A
		@test !(B === A)
	end

	@testset "throw" begin
		@test_throws String A.bug
		@test_throws DimensionMismatch LinearMapAA(L ; idim=(0,0))
		@test_throws String LinearMapAA(L, (_prop=0,))
	end

	@testset "transpose" begin
		@test Matrix(A)' == Matrix(A')
		@test Matrix(A)' == Matrix(transpose(A))
	end

	@testset "wrap" begin # WrappedMap vs FunctionMap
		M1 = rand(3,2)

		A1w = LinearMapAA(M1)
		A1f = LinearMapAA(x -> M1*x, y -> M1'y, size(M1), T=eltype(M1))
		@test Matrix(A1w) == M1
		@test Matrix(A1f) == M1
	end

	@testset "getindex" begin
		@test LinearMapAA_test_getindex(A)
	end

	Ao = LinearMapAA(A._lmap ; odim=(1,size(A,1)), idim=(size(A,2),1))
	@testset "vmul" begin
		@test LinearMapAA_test_vmul(A)
		@test LinearMapAA_test_vmul(A*A'*A) # CompositeMap
		@test LinearMapAA_test_vmul(Ao) # AO type
		@test ndims(Ao) == 2
	end

	@testset "cat" begin
		@test LinearMapAA_test_cat(A)
		@test LinearMapAA_test_cat(Ao)
	end

	@testset "setindex" begin
		@test LinearMapAA_test_setindex(A)
	end

	# add / subtract
	@testset "add" begin
		@test 2A + 6A isa LinearMapAX
		@test 7A - 2A isa LinearMapAX
		@test Matrix(2A + 6A) == 8 * Matrix(A)
		@test Matrix(7A - 2A) == 5 * Matrix(A)
		@test Matrix(7A - 2*ones(size(A))) == 7 * Matrix(A) - 2*ones(size(A))
		@test Matrix(3*ones(size(A)) - 5A) == 3*ones(size(A)) - 5 * Matrix(A)
		@test_throws String @show A + redim(A ; idim=(3,2)) # mismatch dim
	end

	# add identity
	@testset "+I" begin
		@test Matrix(A'A - 7I) == Matrix(A'A) - 7I
	end

	# multiply with identity
	@testset "*I" begin
		@test Matrix(A * 6I) == 6 * Matrix(A)
		@test Matrix(7I * A) == 7 * Matrix(A)
		@test Matrix((false*I) * A) == zeros(size(A))
		@test Matrix(A * (false*I)) == zeros(size(A))
		@test 1.0I * A === A
		@test A * 1.0I === A
		@test I * A === A
		@test A * I === A
	end

	# multiply
	@testset "*" begin
		D = A * A'
		@test D isa LinearMapAX
		@test Matrix(D) == Lm * Lm'
		@test issymmetric(D) == true
		E = A * Lm'
		@test E isa LinearMapAX
		@test Matrix(E) == Lm * Lm'
		F = Lm' * A
		@test F isa LinearMapAX
		@test Matrix(F) == Lm' * Lm
		@test LinearMapAA_test_getindex(F)

		@test LinearMapAA_test_mul()
	end


	# non-adjoint version
	@testset "non-adjoint" begin
		Af = LinearMapAA(forw, (M, N))
		@test Matrix(Af) == Lm
		@test LinearMapAA_test_getindex(Af)
		@test LinearMapAA_test_setindex(Af)
	end

	# kron
	@testset "kron" begin
		@test LinearMapAA_test_kron()
	end

	# FunctionMap for multi-dimensional AO
	@testset "AO FunctionMap" begin
		forw = x -> [cumsum(x; dims=2); zeros(2,size(x,2))]
		back = y -> reverse(cumsum(reverse(y[1:(end-2),:]; dims=2); dims=2); dims=2)
		A = LinearMapAA(forw, (4*3, 2*3) ; idim=(2,3), odim=(4,3))
		@test A isa LinearMapAO
		A = LinearMapAA(forw, back, (4*3, 2*3) ; idim=(2,3), odim=(4,3))
		@test A isa LinearMapAO
		@test Matrix(A') == Matrix(A)'

		A = undim(A) # ensure that undim works
		@test A isa LinearMapAM
		@test Matrix(A') == Matrix(A)'
	end

	true
end
