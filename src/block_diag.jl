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
using Test: @test, @testset

all_ao(As::LinearMapAX...) = all(map(A -> A isa LinearMapAO, As))
same_idim(As::LinearMapAX...) = all(map(A -> A._idim == As[1]._idim, As))
same_odim(As::LinearMapAX...) = all(map(A -> A._odim == As[1]._odim, As))

"""
    B = block_diag(As::LinearMapAX... ; tryop::Bool)
Make block diagonal `LinearMapAX` object from blocks.

Return a `LinearMapAM` unless `tryop` and all blocks have same `idim` and `odim`
Default for `tryop` is `true` if all blocks are type `LinearMapAO`
"""
function block_diag(As::LinearMapAX... ; tryop::Bool = all_ao(As...))
    B = LinearMaps.blockdiag(map(A -> A._lmap, As)...)
    prop = (nblock = length(As),)
    nblock = length(As)
    if (tryop && nblock > 1 && same_idim(As...) && same_odim(As...)) # operator version
        return LinearMapAA(B ; prop=prop,
            idim = (As[1]._idim..., nblock),
            odim = (As[1]._odim..., nblock),
        )
    end
    return LinearMapAA(B, prop)
end


"""
    block_diag(:test)
self test
"""
function block_diag(test::Symbol)
    !(test === :test) && throw("error $test")

    M1 = rand(3,2)
    M2 = rand(5,4)
    M = [M1 zeros(size(M1,1),size(M2,2));
        zeros(size(M2,1),size(M1,2)) M2]

    A1 = LinearMapAA(M1) # WrappedMap
    A2 = LinearMapAA(x -> M2*x, y -> M2'y, size(M2), T=eltype(M2)) # FunctionMap
    B = block_diag(A1, A2)

    @testset "block_diag for AM" begin
        @test Matrix(B) == M
        @test Matrix(B)' == Matrix(B')
    end

    # test LMAO
    @testset "block_diag for AO" begin
        Ao = LinearMapAA(M2 ; odim=(1,5))
        @test block_diag(A1, A1) isa LinearMapAM
        @test block_diag(A1, A1 ; tryop=true) isa LinearMapAO
        @test block_diag(A1, Ao) isa LinearMapAM
        Bo = block_diag(Ao, Ao)
        @test Bo isa LinearMapAO

        Md = blockdiag(sparse(M2), sparse(M2))
        X = rand(4,2)
        Yo = Bo * X
        Yd = Md * vec(X)
        @test vec(Yo) ≈ Yd

        @test Matrix(Bo)' == Matrix(Bo')
    end

    true
end
