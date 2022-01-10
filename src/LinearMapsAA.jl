"""
`module LinearMapsAA`

Provides `AbstractArray` (actually `AbstractMatrix`)
or "Ann Arbor" version of `LinearMap` objects
"""
module LinearMapsAA

using Requires: @require

export LinearMapAA, LinearMapAM, LinearMapAO, LinearMapAX

Indexer = AbstractVector{Int}

include("types.jl")
include("multiply.jl")
include("ambiguity.jl")
include("kron.jl")
include("cat.jl")
include("getindex.jl")
#include("setindex.jl")
include("block_diag.jl")
include("lm-aa.jl")
include("identity.jl")

# support LinearOperators iff user has loaded that package
function __init__()
    @require LinearOperators = "5c8ed15e-5a4c-59e4-a42b-c7e8811fb125" include("wrap-linop.jl")
end

end # module
