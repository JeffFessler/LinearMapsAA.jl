# cat.jl
# test

using LinearMaps: LinearMap
using LinearMapsAA: LinearMapAA, LinearMapAM, LinearMapAO, LinearMapAX
using LinearMapsAA: redim, undim
using LinearAlgebra: I
using Test: @test, @testset

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

    @testset "cat OI" begin
        @test [A I] isa LinearMapAM
        @test [A; I] isa LinearMapAM
        @test [A A; I I] isa LinearMapAM
    end

    @testset "cat AO mixed" begin
        B = redim(A, idim=(1,A._idim...)) # force incompatible dim
        @test [A B] isa LinearMapAM
        @test [A; B] isa LinearMapAM
        Z = [A undim(A) I A._lmap] # Matrix(B)]
        @test Z isa LinearMapAM
        @test Z.hcat == "OAIL"
    end

    true
end


N = 6; M = 8 # non-square to stress test
forw = x -> [cumsum(x); 0; 0]
back = y -> reverse(cumsum(reverse(y[1:N])))
A = LinearMapAA(forw, back, (M, N))
@test LinearMapAA_test_cat(A)

forw = x -> [cumsum(x ; dims=2) [0] [0]]
back = y -> reverse(cumsum(reverse(y[[1],1:N] ; dims=2) ; dims=2) ; dims=2)
O = LinearMapAA(forw, back, (M, N) ; idim=(1,N), odim=(1,M))
@test LinearMapAA_test_cat(O)
