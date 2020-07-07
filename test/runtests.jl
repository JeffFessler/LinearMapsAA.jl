# test/runtests.jl

using LinearMapsAA
using Test: @test, @testset, detect_ambiguities

include("multiply.jl")

@testset "kron" begin
	include("kron.jl")
end

@testset "cat" begin
	include("cat.jl")
end

@testset "getindex" begin
	include("getindex.jl")
end

@testset "setindex" begin
	include("setindex.jl")
end

@testset "block_diag" begin
	include("block_diag.jl")
end

@testset "tests" begin
	include("tests.jl")
end

@test length(detect_ambiguities(LinearMapsAA)) == 0
