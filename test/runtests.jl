# test/runtests.jl

using LinearMapsAA
using Test: @test, @testset, detect_ambiguities

include("multiply.jl")

list = [
"identity"
"kron"
"cat"
"getindex"
"setindex"
"block_diag"
"tests"
]

for file in list
	@testset "$file" begin
		include("$file.jl")
	end
end

@test length(detect_ambiguities(LinearMapsAA)) == 0
