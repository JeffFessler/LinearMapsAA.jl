# test/runtests.jl

using LinearMapsAA: LinearMapsAA
using Test: @test, @testset, @test_broken, detect_ambiguities

function test_ambig(str::String)
    @testset "ambiguities-$str" begin
        tmp = detect_ambiguities(LinearMapsAA)
    #   @show tmp
        @test length(tmp) == 0
    end
end

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
]

for file in list
    @testset "$file" begin
        include("$file.jl")
    end
end

test_ambig("before cuda")

#=
using CUDA causes an import of StaticArrays that leads to a method ambiguity
=#
@testset "cuda" begin
    include("cuda.jl")
end

@testset "ambiguities" begin
    tmp = detect_ambiguities(LinearMapsAA)
    @show tmp # todo
    @test_broken length(tmp) == 0
end
