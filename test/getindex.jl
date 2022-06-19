# getindex.jl
# test

using LinearMapsAA: LinearMapAA, LinearMapAX
using Test: @test, @test_throws, @testset


_slice(a, b) = (a == :) || (b == :) # only slicing supported as of LM 3.7 !


"""
    LinearMapAA_test_getindex(A::LinearMapAX)
tests for `getindex`
"""
function LinearMapAA_test_getindex(A::LinearMapAX)
    B = Matrix(A)
    @test all(size(A) .>= (4,4)) # required by tests

    tf1 = [false; trues(size(A,1)-1)]
    tf2 = [false; trues(size(A,2)-2); false]
    ii1 = (3, 2:4, [2,4], :, tf1)
    ii2 = (2, 3:4, [1,4], :, tf2)
    for i2 in ii2, i1 in ii1
        if _slice(i1, i2)
            @test B[i1,i2] == A[i1,i2]
        else
            @test_throws ErrorException B[i1,i2] == A[i1,i2]
        end
    end

    L = A._lmap
    test_adj = !((:fc in propertynames(L)) && isnothing(L.fc))
    if test_adj
        for i1 in ii2, i2 in ii1
            if _slice(i1, i2)
                @test B'[i1,i2] == A'[i1,i2]
            else
                @test_throws ErrorException B'[i1,i2] == A'[i1,i2]
            end
        end
    end
    true
end


function LinearMapAA_test_getindex_scalar(A::LinearMapAX)
    # "end"
    @test B[3,end-1] == A[3,end-1]
    @test B[end-2,3] == A[end-2,3]
    if test_adj
        @test B'[3,end-1] == A'[3,end-1]
    end

    # [?]
    @test B[1] == A[1]
    @test B[7] == A[7]
    if test_adj
        @test B'[3] == A'[3]
    end
    @test B[end] == A[end] # lastindex

    kk = [k in [3,5] for k = 1:length(A)] # Bool
    @test B[kk] == A[kk]

    # Some tests could rely on the fact that LinearMapAM <: AbstractMatrix,
    # by inheriting from general Base.getindex, but all are provided here.
    @test B[[1, 3, 4]] == A[[1, 3, 4]]
    @test B[4:7] == A[4:7]

    true
end


N = 6; M = 8 # non-square to stress test
forw = x -> [cumsum(x); 0; 0]
back = y -> reverse(cumsum(reverse(y[1:N])))
A = LinearMapAA(forw, back, (M, N))
@test LinearMapAA_test_getindex(A)
# @test LinearMapAA_test_getindex_scalar(A) # no!
