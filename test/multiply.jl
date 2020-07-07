# multiply.jl

#export LinearMapAA_test_vmul # testing
#export LinearMapAA_test_mul # testing
using LinearMapsAA
using Test: @test, @testset


"""
    LinearMapAA_test_vmul(A::LinearMapAM)
    LinearMapAA_test_vmul(A::LinearMapAO)
tests for multiply by vector, scalar, and Matrix
"""
function LinearMapAA_test_vmul(A::LinearMapAM)
    B = Matrix(A)
    (M,N) = size(A)

    u = rand(M)
    v = rand(N)

    @testset "A*v" begin
        Bv = B * v
        Bpu = B' * u

        y = A * v
        x = A' * u
        @test isapprox(Bv, y)
        @test isapprox(Bpu, x)

        mul!(y, A, v)
        mul!(x, A', u)
        @test isapprox(Bv, y)
        @test isapprox(Bpu, x)
    end

    #= nah
    @testset "u'*A" begin
        uB = u' * B
        x = u' * A
        @test isapprox(uB, x)
        mul!(x, u', A)
        @test isapprox(uB, x)
    end
    =#

    @testset "A*X" begin
        X = rand(N,2)
        BX = B * X
        Y = A * X
        @test Y isa LinearMapAM
        Y = Matrix(Y)
        @test isapprox(BX, Y)
        Y .= 0
        # mul!(Y, A, X) # doesn't work because A*X is AM
        # @test isapprox(BX, Y)
    end

    @testset "X*A" begin
        X = rand(2,M)
        XB = X * B
        Y = X * A
        @test Y isa LinearMapAM
        Y = Matrix(Y)
        @test isapprox(XB, Y)
        Y .= 0
        # mul!(Y, X, A) # doesn't work because X*A is AM
        # @test isapprox(XB, Y)
    end

    @testset "5-arg" begin
        y1 = randn(M)
        y2 = copy(y1)
        mul!(y1, A, v, 2, 3)
        mul!(y2, B, v, 2, 3)
        @test isapprox(y1, y2)

        x1 = randn(N)
        x2 = copy(x1)
        mul!(x1, A', u, 4, 3)
        mul!(x2, B', u, 4, 3)
        @test isapprox(x1, x2)
    end

    @testset "*s" begin
        s = 4.2
        C = s * A
        @test isapprox(Matrix(C), s * B)
        C = A * s
        @test isapprox(Matrix(C), B * s)
    end

#=
    s = 4.2
    C = copy(A)
    lmul!(s, C)
    @test isapprox(s * B * v, C * v)

    C = copy(A)
    rmul!(C, s)
    @test isapprox(s * B * v, C * v)
=#

    true
end


function LinearMapAA_test_vmul(A::LinearMapAO)
    B = Matrix(A)
    (M,N) = size(A)

    u = rand(M)
    v = rand(N)
    u = reshape(u, A._odim)
    v = reshape(v, A._idim)

    @testset "A*v" begin
        Bv = reshape(B * v[:], A._odim)
        Bpu = reshape(B' * u[:], A._idim)

        y = A * v
        x = A' * u
        @test isapprox(Bv, y)
        @test isapprox(Bpu, x)

        mul!(y, A, v)
        mul!(x, A', u)
        @test isapprox(Bv, y)
        @test isapprox(Bpu, x)

        A1 = redim(A ; idim = (N,))
        y1 = A1 * vec(v)
        @test y1 ≈ y
    end

    #= nah
    @testset "u'*A" begin
        uB = reshape(u[:]' * B, A._idim)
        x = u' * A
        @test isapprox(uB, x)
        mul!(x, u', A)
        @test isapprox(uB, x)
    end
    =#

    @testset "A*X" begin
        K = 2
        X = rand(A._idim..., K)
        BX = reshape(B * reshape(X, :, K), A._odim..., K)
        Y = A * X
        @test Y isa AbstractArray
        @test isapprox(BX, Y)

        Y .= 0
        mul!(Y, A, X)
        @test isapprox(BX, Y)

        # 5-arg
        Y1 = copy(Y)
        Y2 = copy(Y)
        mul!(Y1, A, X, 2, 3)
        mul!(reshape(Y2,:,K), B, reshape(X,:,K), 2, 3)
        Y2 = reshape(Y2, A._odim..., K)
        @test isapprox(Y1, Y2)
    end

    # todo: should eventually test complex X*A as well

    @testset "X*A" begin
        K = 2
        X = rand(A._odim..., K)
        XB = reshape(X, :, K)' * B
        XB = reshape(XB', A._idim..., K)
        Y = X * A
        @test Y isa AbstractArray
        @test isapprox(XB, Y)

        Y .= 0
        mul!(Y, X, A)
        @test isapprox(XB, Y)

        # 5-arg
        Y1 = copy(Y)
        Y2 = copy(Y)
        mul!(Y1, X, A, 2, 3)
           mul!(reshape(Y2,:,K), B', reshape(X,:,K), 2, 3)
        Y2 = reshape(Y2, A._idim..., K)
           @test isapprox(Y1, Y2)
    end

    @testset "*s" begin
        s = 4.2
        C = s * A
        @test isapprox(Matrix(C), s * B)
        C = A * s
        @test isapprox(Matrix(C), B * s)
    end

    true
end


# object multiplication
function LinearMapAA_test_mul( ;
    A::LinearMapAO = LinearMapAA(rand(6,4) ; odim=(2,3), idim=(1,4)),
)
    (M,N) = size(A)

    @testset "O*O" begin
        @test A'*A isa LinearMapAO
    end

    @testset "O*M" begin
        B = LinearMapAA(rand(N,3)) # AM
        O = redim(A ; idim=(N,))
        @test O*B isa LinearMapAO
    end

    @testset "M*O" begin
        B = LinearMapAA(rand(3,M)) # AM
        O = redim(A ; odim=(M,))
        @test B*O isa LinearMapAO
    end

    true
end
