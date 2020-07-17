"""
`module LinearMapsAA`

Provides `AbstractArray` (actually `AbstractMatrix`)
or "Ann Arbor" version of `LinearMap` objects
"""
module LinearMapsAA

export LinearMapAA, LinearMapAM, LinearMapAO, LinearMapAX

Indexer = AbstractVector{Int}

include("types.jl")
include("multiply.jl")
include("kron.jl")
include("cat.jl")
include("getindex.jl")
include("setindex.jl")
include("block_diag.jl")
include("lm-aa.jl")
include("identity.jl")

end # module
