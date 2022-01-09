# wrap-linop.jl
# Wrap a LinearMapAA around a LinearOperator

export LinearMapAA

using .LinearOperators: LinearOperator


"""
    A = LinearMapAA(L::LinearOperator ; ...)

Wrap a `LinearOperator` in a `LinearMapAX`

# options
- `prop::NamedTuple = NamedTuple()`
- `T = eltype(L)`
- `idim::Dims = (size(L,2),)`
- `odim::Dims = (size(L,1),)`
- `operator::Bool` by default: `false` if both `idim` & `odim` are 1D.

`prop` cannot include the fields `_lmap`, `_prop`, `_idim`, `_odim`

Output `A` is `LinearMapAO` if `operator` is `true`, else `LinearMapAM`.
"""
function LinearMapAA(L::LinearOperator ;
    prop::NamedTuple = NamedTuple(),
    T::Type = eltype(L),
    idim::Dims{Di} = (size(L,2),),
    odim::Dims{Do} = (size(L,1),),
    operator::Bool = length(idim) > 1 || length(odim) > 1,
) where {Di,Do}

    size(L,2) == prod(idim) ||
        throw(DimensionMismatch("size2=$(size(L,2)) vs idim=$idim"))
    size(L,1) == prod(odim) ||
        throw(DimensionMismatch("size1=$(size(L,1)) vs odim=$odim"))
    length(intersect(propertynames(prop), LMAAkeys)) > 0 &&
        throw("invalid property field among $(propertynames(prop))")

    forw!(y, x) = mul!(y, L, x)
    back!(x, y) = mul!(x, L', y)
    return operator ?
         LinearMapAO{T,Do,Di}(forw!, back!, prop, idim, odim) :
         LinearMapAM{T,Do,Di}(forw!, back!, prop, idim, odim)
end
