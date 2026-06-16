using LinearMapsAA: LinearMapsAA
import Aqua
using Test: @testset

@testset "aqua" begin
    Aqua.test_all(LinearMapsAA)
end
