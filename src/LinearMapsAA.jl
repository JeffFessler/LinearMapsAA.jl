"""
`module LinearMapsAA`

Provides `AbstractArray` (actually `AbstractMatrix`)
or "Ann Arbor" version of `LinearMap` objects
"""
module LinearMapsAA

export LinearMapAA

# Indexer = AbstractVector{Int}

include("lm-aa.jl")
include("block_diag.jl")

end # module
