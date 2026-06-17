using LinearMapsAA: LinearMapsAA
import Aqua
using Test: @testset

@testset "aqua" begin
    Aqua.test_all(
        LinearMapsAA;
        deps_compat = (; ignore = [:LinearAlgebra, :SparseArrays]),
        piracies = false, # todo: see src/identity
    )
end
