# tests

using LinearMaps: LinearMap
using LinearMapsAA: LinearMapAA, LinearMapAM, LinearMapAO, LinearMapAX
using LinearAlgebra: issymmetric, ishermitian, isposdef, I
using SparseArrays: sparse
using Test: @test, @testset, @test_throws

#B = 1:6
#L = LinearMap(x -> B*x, y -> B'*y, 6, 1)

B = reshape(1:6, 6, 1)
@test Matrix(LinearMapAA(B)) == B

# ensure that "show" is concise even for big `prop`
L = LinearMapAA(LinearMap(ones(3,4)), (a=1:3, b=ones(99,99)))
show(isinteractive() ? stdout : devnull, L)

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
    @test :name ∈ propertynames(A)
    @test A._prop == prop
    @test A.name == prop.name
    @test eltype(A) == eltype(L)
    @test Base.eltype(A) == eltype(L) # codecov
    @test ndims(A) == 2
    @test size(A) == size(L)
    @test redim(A) isa LinearMapAX
end

@testset "deepcopy" begin
    B = deepcopy(A)
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

Ao = LinearMapAA(A._lmap ; odim=(1,size(A,1)), idim=(size(A,2),1))
@testset "vmul" begin
    @test LinearMapAA_test_vmul(A)
    @test LinearMapAA_test_vmul(A*A'*A) # CompositeMap
    @test LinearMapAA_test_vmul(Ao) # AO type
    @test ndims(Ao) == 2
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

# AO FunctionMap complex
@testset "AO FM C" begin
    T = ComplexF16
    c = T(2im)
    forw! = (y,x) -> copyto!(y,x) .*= c
    back! = (x,y) -> copyto!(x,y) .*= conj(c)
    dims = (2,3)
    O = LinearMapAA(forw!, back!, (1,1).*prod(dims) ;
        T=T, idim=dims, odim=dims)
    x = rand(T, dims)
    @test O*x == c*x
    @test O'*x == conj(c)*x
    @test Matrix(O') == Matrix(O)'
end

# non-adjoint version
@testset "non-adjoint" begin
    Af = LinearMapAA(forw, (M, N))
    @test Matrix(Af) == Lm
    @test LinearMapAA_test_getindex(Af)
#   @test LinearMapAA_test_setindex(Af)
end

@testset "AO for 1D" begin
    B = LinearMapAO(A)
    @test B isa LinearMapAO
    X = rand(N,2)
    Y = B * X
    @test Y isa AbstractArray
    @test Y ≈ Lm * X
    Z = B' * Y
    @test Z isa AbstractArray
    @test Z ≈ Lm' * Y
end

# FunctionMap for multi-dimensional AO
@testset "AO FM 2D" begin
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
