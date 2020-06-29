# test/runtests.jl

using LinearMapsAA
using Test: @test, @testset, detect_ambiguities

@test LinearMapAA(:test)

@testset "block_diag" begin
    @test block_diag(:test)
end

@test length(detect_ambiguities(LinearMapsAA)) == 0
