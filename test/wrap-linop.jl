# wrap-linop.jl
# test wrapping of a LinearMapAX in a LinearOperator and vice-versa

using LinearMapsAA: LinearMapAA, LinearMapAM, LinearOperator_from_AA
using LinearOperators: LinearOperator
using Test: @test, @testset

forw! = cumsum!
back! = (x, y) -> reverse!(cumsum!(x, reverse!(copyto!(x, y))))

N = 9
T = Float32
L = LinearOperator(
    T, N, N, false, false, forw!,
    nothing, # transpose mul!
    back!,
)

x = rand(N)
y = L * x
@test y == cumsum(x)
@test Matrix(L)' == Matrix(L')

A = LinearMapAA(L) # wrap LinearOperator
@test A isa LinearMapAM
@test Matrix(A)' == Matrix(A')
@test A * x == cumsum(x)

B = LinearMapAA(forw!, back!, (N, N); T)
L = LinearOperator_from_AA(B) # wrap LinearMapAM
@test L isa LinearOperator
@test Matrix(L)' == Matrix(L')
@test L * x == cumsum(x)
