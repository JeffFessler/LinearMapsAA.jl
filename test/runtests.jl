# test/runtests.jl

using LinearMapsAA
using Test: @test, detect_ambiguities

@test LinearMapAA(:test)

@test length(detect_ambiguities(LinearMapsAA)) == 0
