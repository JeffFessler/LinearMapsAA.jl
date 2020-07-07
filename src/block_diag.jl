#=
block_diag.jl
Like SparseArrays:blockdiag but slightly different name, just to be safe,
because SparseArrays are also an AbstractArray type.

Motivated by compressed sensing dynamic MRI
where each frame has different k-space sampling
2020-06-08 Jeff Fessler
=#

export block_diag
import LinearMaps
using SparseArrays: blockdiag, sparse

same_idim(As::LinearMapAX...) = all(map(A -> A._idim == As[1]._idim, As))
same_odim(As::LinearMapAX...) = all(map(A -> A._odim == As[1]._odim, As))

"""
    B = block_diag(As::LinearMapAX... ; tryop::Bool)
Make block diagonal `LinearMapAX` object from blocks.

Return a `LinearMapAM` unless `tryop` and all blocks have same `idim` and `odim`.

Default for `tryop` is `true` if all blocks are type `LinearMapAO`.
"""
block_diag(As::LinearMapAO... ; tryop::Bool = true) = block_diag(tryop, As...)

block_diag(As::LinearMapAX... ; tryop::Bool = false) = block_diag(tryop, As...)

function block_diag(tryop::Bool, As::LinearMapAX...)
    B = LinearMaps.blockdiag(map(A -> A._lmap, As)...)
    nblock = length(As)
    prop = (nblock = nblock,)
    if (tryop && nblock > 1 && same_idim(As...) && same_odim(As...)) # operator version
        return LinearMapAA(B ; prop=prop,
            idim = (As[1]._idim..., nblock),
            odim = (As[1]._odim..., nblock),
        )
    end
    return LinearMapAA(B, prop)
end
