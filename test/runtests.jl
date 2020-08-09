# test/runtests.jl

using LinearMapsAA
using Test: @test, @testset, detect_ambiguities


list = [
"identity"
"multiply"
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
