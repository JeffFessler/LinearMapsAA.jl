#=
ambiguity.jl
Avoiding method ambiguities
This may be a bit of a whack-a-mole™ exercise...
=#

import LinearAlgebra

const Trans = LinearAlgebra.Transpose{<: Any, <: LinearAlgebra.RealHermSymComplexSym}
const Adjoi = LinearAlgebra.Adjoint{<: Any, <: LinearAlgebra.RealHermSymComplexHerm}

Base.:(*)(A::LinearMapAM, D::LinearAlgebra.Diagonal) = AM_M(A, D)
Base.:(*)(D::LinearAlgebra.Diagonal, B::LinearMapAM,) = M_AM(D, B)

Base.:(*)(A::LinearMapAM, B::LinearAlgebra.AbstractTriangular) = AM_M(A, B)
Base.:(*)(A::LinearAlgebra.AbstractTriangular, B::LinearMapAM) = M_AM(A, B)

Base.:(*)(A::LinearMapAM, B::Trans) = AM_M(A, B)
Base.:(*)(A::Trans, B::LinearMapAM) = M_AM(A, B)

Base.:(*)(A::LinearMapAM, B::Adjoi) = AM_M(A, B)
Base.:(*)(A::Adjoi, B::LinearMapAM) = M_AM(A, B)

# see https://github.com/Jutho/LinearMaps.jl/issues/118
Base.:(*)(A::LinearMapAM,
   B::LinearAlgebra.Adjoint{<: Any, <: LinearAlgebra.AbstractRotation}) =
   throw("AbstractRotation lacks size so * is unsupported")
#Base.:(*)(A::Given, B::LinearMapAM) = M_AM(A, B)

Base.:(*)(x::LinearAlgebra.AdjointAbsVec, A::LinearMapAM) = (A' * x')'
Base.:(*)(x::LinearAlgebra.TransposeAbsVec, A::LinearMapAM) =
    transpose(transpose(A) * transpose(x))
