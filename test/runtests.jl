# test/runtests.jl

using LinearMapsAA
using Test: @test, @testset, @test_broken, detect_ambiguities

include("multiply.jl")

list = [
"ambiguity"
"identity"
"kron"
"cat"
"getindex"
#"setindex"
"block_diag"
"tests"
"wrap-linop"
"cuda"
]

for file in list
    @testset "$file" begin
        include("$file.jl")
    end
end

@testset "ambiguities" begin
    tmp = detect_ambiguities(LinearMapsAA)
@show tmp # todo
    @test_broken length(tmp) == 0
end
