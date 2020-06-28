#=
cat.jl
Concatenation support for LinearMapAX
2018-01-19, Jeff Fessler, University of Michigan
=#

export lmaa_hcat, lmaa_vcat, lmaa_hvcat

export LinearMapAA_test_cat # testing

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
LMcat{T} = Union{LinearMapAM{T}, LinearMap{T}, UniformScaling{T}} # settle
#LMelse{T} = Union{LinearMap{T},UniformScaling{T},AbstractMatrix{T}} # non AM

# convert to something suitable for LinearMap.*cat
function lm_promote(A::LMcat)
#    @show typeof(A)
    A isa LinearMapAM ? A._lmap :
    A isa UniformScaling ? A : # leave unchanged - ok for LinearMaps.*cat
    A # otherwise it is this
#    throw("bug") # should only be only of LMcat types
#    A isa AbstractMatrix ? LinearMap(A) :
#    A isa LinearMap ?
end

# single-letter codes for cat objects, e.g., [A I A] becomes "AIA"
#= not so useful
function lm_code(A)
    isa(A, LinearMapAM) ? "A" :
    isa(A, AbstractMatrix) ? "M" :
    isa(A, UniformScaling) ? "I" :
    isa(A, LinearMap) ? "L" :
    "?"
end
=#

# concatenate the single-letter codes, e.g., [A I A] becomes "AIA"
# lm_name = As -> *(lm_code.(As)...)
lm_name = As -> nothing

# these rely on LinearMap.*cat methods
"`B = lmaa_hcat(A1, A2, ...)` `hcat` of multiple objects"
# todo: check odim ...
lmaa_hcat(As::LMcat...) =
    LinearMapAA(hcat(lm_promote.(As)...), (hcat=lm_name(As),))
"`B = lmaa_vcat(A1, A2, ...)` `vcat` of multiple objects"
lmaa_vcat(As::LMcat...) =
    LinearMapAA(vcat(lm_promote.(As)...), (vcat=lm_name(As),))
"`B = lmaa_hvcat(rows, A1, A2, ...)` `hvcat` of multiple objects"
lmaa_hvcat(rows::NTuple{nr,Int} where {nr}, As::LMcat...) =
    LinearMapAA(hvcat(rows, lm_promote.(As)...), (hvcat=lm_name(As),))

# a single leading LinearMapAM followed by others is clear
Base.hcat(A1::LinearMapAM, As::LMcat...) = # todo
    lmaa_hcat(A1, As...)
Base.vcat(A1::LinearMapAM, As::LMcat...) =
    lmaa_vcat(A1, As...)
Base.hvcat(rows::NTuple{nr,Int} where {nr}, A1::LinearMapAM, As::LMcat...) =
    lmaa_hvcat(rows, A1, As...)
# or in 2nd position, beyond that, user can use lmaa_*
#Base.hcat(A1::LMelse, A2::LinearMapAM, As::LMcat...) = # fails!?
Base.hcat(A1::UniformScaling, A2::LinearMapAM, As::LMcat...) =
    lmaa_hcat(A1, A2, As...)
Base.vcat(A1::UniformScaling, A2::LinearMapAM, As::LMcat...) =
    lmaa_vcat(A1, A2, As...)
Base.hvcat(rows::NTuple{nr,Int} where nr,
        A1::UniformScaling, A2::LinearMapAM, As::LMcat...) =
    lmaa_hvcat(rows, A1, A2, As...)


# special handling for AO

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
`hvcat` that discards special `idim` and `odim` information (too hard!) # todo
"""
Base.hvcat(rows::NTuple{nr,Int} where nr,
        A1::LinearMapAO, As::LinearMapAO...) =
    lmaa_hvcat(rows, undim(A1), undim.(As)...)


# test

#=
    this approach using eval() works only in the global scope!

    N = 6
    forw = x -> [cumsum(x); 0] # non-square to stress test
    back = y -> reverse(cumsum(reverse(y[1:N])))
    prop = (name="cumsum", extra=1)
    A = LinearMapAA(forw, back, (N+1, N), prop)

    list1 = [
        :([A I]), :([I A]), :([2*A 3*A]),
        :([A; I]), :([I; A]), :([2*A; 3*A]),
        :([A A I]), :([A I A]), :([2*A 3*A 4*A]),
        :([A; A; I]), :([A; I; A]), :([2*A; 3*A; 4*A]),
        :([I A I]), :([I A A]),
        :([I; A; I]), :([I; A; A]),
    #   :([I I A]), :([I; I; A]), # unsupported
        :([A A; A A]), :([A I; 2A 3I]), :([I A; 2I 3A]),
    #   :([I A; A I]), :([A I; I A]), # weird sizes
        :([A I A; 2A 3I 4A]), :([I A I; 2I 3A 4I]),
    #   :([A I A; I A A]), :([I A 2A; 3A 4I 5A]), # weird
    #   :([I I A; I A I]), # unsupported
        :([A A I; 2A 3A 4I]),
        :([A I I; 2A 3I 4I]),
    ]

    M = Matrix(A)
    list2 = [
        :([M I]), :([I M]), :([2*M 3*M]),
        :([M; I]), :([I; M]), :([2*M; 3*M]),
        :([M M I]), :([M I M]), :([2*M 3*M 4*M]),
        :([M; M; I]), :([M; I; M]), :([2*M; 3*M; 4*M]),
        :([I M I]), :([I M M]),
        :([I; M; I]), :([I; M; M]),
    #   :([I I M]), :([I; I; M]), # unsupported
        :([M M; M M]), :([M I; 2M 3I]), :([I M; 2I 3M]),
    #   :([I M; M I]), :([M I; I M]), # weird sizes
        :([M I M; 2M 3I 4M]), :([I M I; 2I 3M 4I]),
    #   :([M I M; I M M]), :([I M 2M; 3M 4I 5M]), # weird
    #   :([I I M; I M I]), # unsupported
        :([M M I; 2M 3M 4I]),
        :([M I I; 2M 3I 4I]),
    ]

    length(list1) != length(list2) && throw("bug")

    for ii in 1:length(list1)
        e1 = list1[ii]
        b1 = eval(e1)
        e2 = list2[ii]
        b2 = eval(e2)
        @test b1 isa LinearMapAX
    end
=#


"""
    LinearMapAA_test_cat(A::LinearMapAM)
test hcat vcat hvcat
"""
function LinearMapAA_test_cat(A::LinearMapAM)
    Lm = LinearMap{eltype(A)}(x -> A*x, y -> A'*y, size(A,1), size(A,2))
    M = Matrix(A)

#    LinearMaps supports *cat of LM and UniformScaling only in v2.6.1
#    but see: https://github.com/Jutho/LinearMaps.jl/pull/71
#    B0 = [M Lm] # fails!

#=
    # cannot get cat with AbstractMatrix to work
    M1 = reshape(1:35, N+1, N-1)
    H2 = [A M1]
    @test H2 isa LinearMapAX
    @test Matrix(H2) == [Matrix(A) H2]
    H1 = [M1 A]
    @test H1 isa LinearMapAX
    @test Matrix(H1) == [M1 Matrix(A)]

    M2 = reshape(1:(3*N), 3, N)
    V1 = [M2; A]
    @test V1 isa LinearMapAX
    @test Matrix(V1) == [M2; Matrix(A)]
    V2 = [A; M2]
    @test V2 isa LinearMapAX
    @test Matrix(V2) == [Matrix(A); M2]
=#

    fun0 = A -> [
        [A I], [I A], [2A 3A],
        [A; I], [I; A], [2A; 3A],
        [A A I], [A I A], [2A 3A 4A],
        [A; A; I], [A; I; A], [2A; 3A; 4A],
        [I A I], [I A A],
        [I; A; I], [I; A; A],
    #   [I I A], [I; I; A], # unsupported
        [A A; A A], [A I; 2A 3I], [I A; 2I 3A],
    #   [I A; A I], [A I; I A], # weird sizes
        [A I A; 2A 3I 4A], [I A I; 2I 3A 4I],
    #   [A I A; I A A], [I A 2A; 3A 4I 5A], # weird
    #   [I I A; I A I], # unsupported
        [A A I; 2A 3A 4I],
        [A I I; 2A 3I 4I],
    #   [A Lm], # need one LinearMap test for codecov (see below)
    ]

    list1 = fun0(A)
    list2 = fun0(M)

    if true # need one LinearMap test for codecov
        push!(list1, [A Lm])
        push!(list2, [M M]) # trick because [M Lm] fails
    end

    for ii in 1:length(list1)
        b1 = list1[ii]
        b2 = list2[ii]
        @test b1 isa LinearMapAX
        @test b2 isa AbstractMatrix
        @test Matrix(b1) == b2
        @test Matrix(b1') == Matrix(b1)'
    end

    true
end


"""
    LinearMapAA_test_cat(A::LinearMapAO)
test hcat vcat hvcat for AO
"""
function LinearMapAA_test_cat(A::LinearMapAO)
    M = Matrix(A)

    @testset "cat AO" begin
        H = [A A]
        V = [A; A]
        B = [A 2A; 3A 4A]
        @test H isa LinearMapAO
        @test V isa LinearMapAO
        @test B isa LinearMapAM # !!
        @test Matrix(H) == [M M]
        @test Matrix(V) == [M; M]
        @test Matrix(B) == [M 2M; 3M 4M]
    end

    true
end
