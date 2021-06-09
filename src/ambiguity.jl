#=
ambiguity.jl
Avoiding method ambiguities
This may be a bit of a whack-a-mole™ exercise...
=#

Base.(*)(A::LinearAlgebra.AbstractTriangular, B::LinearMapAM) = M_AM(A, B)
Base.(*)(transA::LinearAlgebra.Transpose, B::LinearMapAM) = M_AM(transA, B)
