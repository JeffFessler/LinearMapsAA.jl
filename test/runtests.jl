# test/runtests.jl

using LinearMapsAA
using Test: @test, @testset, detect_ambiguities

@testset "LinearMapAA" begin
	@test LinearMapAA(:test)
end

@testset "block_diag" begin
	@test block_diag(:test)
end

@test length(detect_ambiguities(LinearMapsAA)) == 0
