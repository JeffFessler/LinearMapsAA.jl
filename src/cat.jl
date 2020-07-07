#=
cat.jl
Concatenation support for LinearMapAX
2018-01-19, Jeff Fessler, University of Michigan
=#

export lmaa_hcat, lmaa_vcat, lmaa_hvcat

using LinearAlgebra: UniformScaling, I


#=
function warndim(A::LinearMapAX)
    ((length(A._idim) > 1) || (length(A._odim) > 1)) && @warn("dim ignored")
    nothing
end
=#


# cat (hcat, vcat, hvcat) are tricky for avoiding type piracy
# It is especially hard to handle AbstractMatrix,
# so typically force the user to wrap it in LinearMap(AX) first.
#LMcat{T} = Union{LinearMapAM{T}, LinearMap{T}, UniformScaling{T},AbstractMatrix{T}}
LMcat{T} = Union{LinearMapAX{T}, LinearMap{T}, UniformScaling{T}} # settle
#LMelse{T} = Union{LinearMap{T},UniformScaling{T},AbstractMatrix{T}} # non AM

# convert to something suitable for LinearMap.*cat
function lm_promote(A::LMcat)
#    @show typeof(A)
    A isa LinearMapAX ? A._lmap :
    A isa UniformScaling ? A : # leave unchanged - ok for LinearMaps.*cat
    A # otherwise it is this
#    throw("bug") # should only be only of LMcat types
#    A isa AbstractMatrix ? LinearMap(A) :
#    A isa LinearMap ?
end

# single-letter codes for cat objects, e.g., [A I A] becomes "AIA"
lm_code(::LinearMapAM) = "A"
lm_code(::LinearMapAO) = "O"
lm_code(::UniformScaling) = "I"
lm_code(::LinearMap) = "L"
#lm_code(::AbstractMatrix) = "M"
#lm_code(::Any) = "?"

# concatenate the single-letter codes, e.g., [A I A] becomes "AIA"
lm_name = As -> *(lm_code.(As)...)
# lm_name = As -> nothing

# these rely on LinearMap.*cat methods
"`B = lmaa_hcat(A1, A2, ...)` `hcat` of multiple objects"
lmaa_hcat(As::LMcat...) =
    LinearMapAA(hcat(lm_promote.(As)...), (hcat=lm_name(As),))
"`B = lmaa_vcat(A1, A2, ...)` `vcat` of multiple objects"
lmaa_vcat(As::LMcat...) =
    LinearMapAA(vcat(lm_promote.(As)...), (vcat=lm_name(As),))
"`B = lmaa_hvcat(rows, A1, A2, ...)` `hvcat` of multiple objects"
lmaa_hvcat(rows::NTuple{nr,Int} where {nr}, As::LMcat...) =
    LinearMapAA(hvcat(rows, lm_promote.(As)...), (hvcat=lm_name(As),))

# a single leading LinearMapAM followed by others is clear
Base.hcat(A1::LinearMapAX, As::LMcat...) =
    lmaa_hcat(A1, As...)
Base.vcat(A1::LinearMapAX, As::LMcat...) =
    lmaa_vcat(A1, As...)
Base.hvcat(rows::NTuple{nr,Int} where {nr}, A1::LinearMapAX, As::LMcat...) =
    lmaa_hvcat(rows, A1, As...)
# or in 2nd position, beyond that, user can use lmaa_*
#Base.hcat(A1::LMelse, A2::LinearMapAM, As::LMcat...) = # fails!?
Base.hcat(A1::UniformScaling, A2::LinearMapAX, As::LMcat...) =
    lmaa_hcat(A1, A2, As...)
Base.vcat(A1::UniformScaling, A2::LinearMapAX, As::LMcat...) =
    lmaa_vcat(A1, A2, As...)
Base.hvcat(rows::NTuple{nr,Int} where nr,
        A1::UniformScaling, A2::LinearMapAX, As::LMcat...) =
    lmaa_hvcat(rows, A1, A2, As...)


# special handling for solely AO collections

function _hcat(tryop::Bool, As::LinearMapAO...)
    B = LinearMaps.hcat(map(A -> A._lmap, As)...)
    nblock = length(As)
    prop = (nblock = nblock,)
    if (tryop && nblock > 1 && same_idim(As...) && same_odim(As...)) # operator version
        return LinearMapAA(B ; prop=prop,
            idim = (As[1]._idim..., nblock),
            odim = As[1]._odim,
        )
    end
    return LinearMapAA(B ; prop=prop)
end

function _vcat(tryop::Bool, As::LinearMapAO...)
    B = LinearMaps.vcat(map(A -> A._lmap, As)...)
    nblock = length(As)
    prop = (nblock = nblock,)
    if (tryop && nblock > 1 && same_idim(As...) && same_odim(As...)) # operator version
        return LinearMapAA(B ; prop=prop,
            idim = As[1]._idim,
            odim = (As[1]._odim..., nblock),
        )
    end
    return LinearMapAA(B ; prop=prop)
end

"""
    hcat(As::LinearMapAO... ; tryop::Bool=true)
`hcat` with (by default) attempt to append `nblock` to `idim` if consistent blocks.
"""
Base.hcat(A1::LinearMapAO, As::LinearMapAO... ; tryop::Bool=true) =
    _hcat(tryop, A1, As...)

"""
    vcat(As::LinearMapAO... ; tryop::Bool=true)
`vcat` with (by default) attempt to append `nblock` to `odim` if consistent blocks.
"""
Base.vcat(A1::LinearMapAO, As::LinearMapAO... ; tryop::Bool=true) =
    _vcat(tryop, A1, As...)

"""
    hvcat(rows, As::LinearMapAO...)
`hvcat` that discards special `idim` and `odim` information (too hard!) # todo?
"""
Base.hvcat(rows::NTuple{nr,Int} where nr,
        A1::LinearMapAO, As::LinearMapAO...) =
    lmaa_hvcat(rows, undim(A1), undim.(As)...)
