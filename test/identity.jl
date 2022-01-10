# identity.jl
# test

using Test: @test, @testset
using LinearAlgebra: I

@testset "identity" begin
    X = rand(2,3,4) # AbstractArray
    @test I * X === X
    @test (5I) * X == 5*X
end
