# wrap-linop.jl

using LinearMapsAA: LinearMapAA, LinearMapAM
using LinearOperators: LinearOperator
using Test: @test, @testset

forw! = cumsum!
back! = (x, y) -> reverse!(cumsum!(x, reverse!(copyto!(x, y))))

N = 9
L = LinearOperator(Float32, N, N, false, false, forw!,
     nothing, # will be inferred
     back!,
)

x = rand(N)
y = L * x
@test y == cumsum(x)

@test Matrix(L)' == Matrix(L')

A = LinearMapAA(L)
@test A isa LinearMapAM

@test Matrix(A)' == Matrix(A')

@test A * x == cumsum(x)


# todo: test the other way
