# setindex.jl
# test setindex! (deprecated)

using LinearMapsAA: LinearMapAA, LinearMapAM
using Test: @test


"""
`LinearMapAA_test_setindex(A::LinearMapAM)`
"""
function LinearMapAA_test_setindex(A::LinearMapAM)

    @test all(size(A) .>= (4,4)) # required by tests

    tf1 = [false; trues(size(A,1)-1)]
    tf2 = [false; trues(size(A,2)-2); false]
    ii1 = (3, 2:4, [2,4], :, tf1)
    ii2 = (2, 3:4, [1,4], :, tf2)

    # test all [?,?] combinations with array "X"
    for i2 in ii2
        for i1 in ii1
            B = deepcopy(A)
            X = 2 .+ A[i1,i2].^2 # values must differ from A[ii,jj]
            B[i1,i2] = X
            Am = Matrix(A)
            Bm = Matrix(B)
            Am[i1,i2] = X
            @test isapprox(Am, Bm)
        end
    end

    # test all [?,?] combinations with scalar "s"
    for i2 in ii2
        for i1 in ii1
            B = deepcopy(A)
            s = maximum(abs.(A[:])) + 2
            B[i1,i2] = s
            Am = Matrix(A)
            Bm = Matrix(B)
            if (i1 == Colon() || ndims(i1) > 0) || (i2 == Colon() || ndims(i2) > 0)
                Am[i1,i2] .= s
            else
                Am[i1,i2] = s
            end
            @test isapprox(Am, Bm)
        end
    end

    # others not supported for now
    set1 = (3, ) # [3], [2,4], 2:4, (1:length(A)) .== 2), end
    for s1 in set1
        B = deepcopy(A)
        X = 2 .+ A[s1].^2 # values must differ from A[ii,jj]
        B[s1] = X
        Am = Matrix(A)
        Bm = Matrix(B)
        Am[s1] = X
        @test isapprox(Am, Bm)
    end

    # insanity below here

    # A[:] = s
    B = deepcopy(A)
    B[:] = 5
    Am = Matrix(A)
    Am[:] .= 5
    Bm = Matrix(B)
    @test Bm == Am

    # A[:] = v
    B = deepcopy(A)
    v = 1:length(A)
    B[:] = v
    Am = Matrix(A)
    Am[:] .= v
    Bm = Matrix(B)
    @test Bm == Am

    # A[:,:]
    B = deepcopy(A)
    B[:,:] = 6
    Am = Matrix(A)
    Am[:,:] .= 6
    Bm = Matrix(B)
    @test Bm == Am

    true
end

N = 6; M = 8 # non-square to stress test
forw = x -> [cumsum(x); 0; 0]
back = y -> reverse(cumsum(reverse(y[1:N])))
A = LinearMapAA(forw, back, (M, N))
@test LinearMapAA_test_setindex(A)
