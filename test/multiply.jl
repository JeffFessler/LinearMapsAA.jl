# multiply.jl

#export LinearMapAA_test_vmul # testing
#export LinearMapAA_test_mul # testing
using LinearMaps: LinearMap
using LinearMapsAA
using LinearAlgebra: I, issymmetric
import LinearAlgebra # Adjoint
using Test: @test, @testset


"""
    LinearMapAA_test_vmul(A::LinearMapAM)
    LinearMapAA_test_vmul(A::LinearMapAO)
tests for multiply by vector, scalar, and Matrix
"""
function LinearMapAA_test_vmul(A::LinearMapAM{T}) where {T}
    B = Matrix(A)
    (M,N) = size(A)

    u = rand(T, M)
    v = rand(T, N)

    @testset "A*v" begin
        y = A * v
        x = A' * u
        @test y ≈ B * v
        @test x ≈ B' * u
        mul!(y, A, v)
        mul!(x, A', u)
        @test y ≈ B * v
        @test x ≈ B' * u
    end

    @testset "u'*A" begin
        x = u' * A
        @test x isa LinearAlgebra.Adjoint
        @test x ≈ u' * B
        mul!(x, u', A)
        @test x isa LinearAlgebra.Adjoint
        @test x ≈ u' * B
        @test (u' * A) * v ≈ u' * (A * v) ≈ u' * (B * v)
    end

    @testset "transpoz(u)*A" begin
        tr = transpose
        x = tr(u) * A
        @test x isa LinearAlgebra.Transpose
        @test x ≈ tr(u) * B
        mul!(x, tr(u), A)
        @test x isa LinearAlgebra.Transpose
        @test x ≈ tr(u) * B
        @test (tr(u) * A) * v ≈ tr(u) * (A * v) ≈ tr(u) * (B * v)
    end

    @testset "A*X" begin
        X = rand(T, N, 2)
        Y = A * X
        @test Y isa LinearMapAM
        @test Matrix(Y) ≈ B * X
        # mul!(Y, A, X) # inapplicable because A*X is AM
    end

    @testset "X*A" begin
        X = rand(T, 2, M)
        Y = X * A
        @test Y isa LinearMapAM
        @test Matrix(Y) ≈ X * B
        # mul!(Y, X, A) # inapplicable because X*A is AM
    end

    @testset "Y'*A" begin
        Y = rand(T, M, 2)
        X = Y' * A
        @test X isa LinearMapAM
        @test Matrix(X) ≈ Y' * B
        # mul!(X, Y', A) # inapplicable because Y'*A is AM
    end

    @testset "5-arg" begin
        y1 = randn(T, M)
        y2 = copy(y1)
        y3 = 2 * B * v + 3 * y1
        mul!(y1, A, v, 2, 3)
        mul!(y2, B, v, 2, 3)
        @test y1 ≈ y2 ≈ y3

        x1 = randn(T, N)
        x2 = copy(x1)
        x3 = 4 * B' * u + 3 * x1
        mul!(x1, A', u, 4, 3)
        mul!(x2, B', u, 4, 3)
        @test x1 ≈ x2 ≈ x3
    end

    @testset "*s" begin
        s = T <: Complex ? 4.2 + 3.7im : 4.2
        C = s * A
        @test Matrix(C) ≈ s * B
        C = A * s
        @test Matrix(C) ≈ B * s
    end

#=
    @testset "lmul! rmul!" begin
        s = 4.2
        C = copy(A)
        lmul!(s, C)
        @test isapprox(s * B * v, C * v)

        C = copy(A)
        rmul!(C, s)
        @test isapprox(s * B * v, C * v)
    end
=#

    true
end


function LinearMapAA_test_vmul(O::LinearMapAO{T}) where {T}
    B = Matrix(O)
    (M,N) = size(O)

    u = randn(T, M)
    v = randn(T, N)
    u = reshape(u, O._odim)
    v = reshape(v, O._idim)

    @testset "O*v" begin
        Bv = reshape(B * vec(v), O._odim)
        Bpu = reshape(B' * vec(u), O._idim)

        y = O * v
        x = O' * u
        @test y ≈ Bv
        @test x ≈ Bpu

        mul!(y, O, v)
        mul!(x, O', u)
        @test y ≈ Bv
        @test x ≈ Bpu

        O1 = redim(O ; idim = (N,))
        y1 = O1 * vec(v)
        @test y1 ≈ y
    end

    @testset "u'*O" begin
        if O._odim == (M,) # makes sense only in this case
            if O._idim == (N,)
                uB = u' * B
            else
                uB = reshape(u' * B, O._idim)
            end
            x = u' * O
            @test x ≈ uB
#display(O)
#@show size(x), size(uB)
#@show x, uB
            mul!(x, u', O)
            @test x ≈ uB
            @test (u' * O) * v ≈ u' * (O * v) ≈ (u' * B) * vec(v)
        end
    end

#=
todo
    @testset "transpoz(u)*O" begin
        if O._odim == (M,) # makes sense only in this case
            uB = reshape(u[:]' * B, O._idim)
        end
    end
=#

    @testset "O*X" begin
        K = 2
        X = randn(T, O._idim..., K)
        BX = reshape(B * reshape(X, :, K), O._odim..., K)
        Y = O * X
        @test Y isa AbstractArray
        @test Y ≈ BX

        Y = similar(Y)
        mul!(Y, O, X)
        @test Y ≈ BX

        # 5-arg
        Y1 = copy(Y)
        Y2 = copy(Y)
        mul!(Y1, O, X, 2, 3)
        mul!(reshape(Y2,:,K), B, reshape(X,:,K), 2, 3)
        Y2 = reshape(Y2, O._odim..., K)
        @test Y1 ≈ Y2
    end

    @testset "X*O" begin
        K = 2
        X = randn(T, O._odim..., K)
        XB = reshape(X, :, K)' * B
        XB = reshape(XB', O._idim..., K)
        Y = X * O
        @test Y isa AbstractArray
        @test Y ≈ XB
        mul!(Y, X, O, 1, 0)
        @test Y ≈ XB

        Y = similar(Y)
        mul!(Y, X, O)
        @test Y ≈ XB

        # 5-arg
        Y1 = copy(Y)
        Y2 = copy(Y)
        mul!(Y1, X, O, 2, 3)
        # todo: is this the appropriate comparison for complex X*O?
        mul!(reshape(Y2,:,K), B', reshape(X,:,K), 2, 3)
        Y2 = reshape(Y2, O._idim..., K)
        @test Y1 ≈ Y2
    end

#= todo
    @testset "Y'*O" begin
        if O._odim == (M,) # makes sense only in this case
            Y = rand(T, M, 2)
@warn("todo")
        end
    end
=#

    @testset "*s" begin
        s = T <: Complex ? 4.2 + 3.7im : 4.2
        C = s * O
        @test Matrix(C) ≈ s * B
        C = O * s
        @test Matrix(C) ≈ B * s
    end

    true
end


# object multiplication
function LinearMapAA_test_mul( ;
    O::LinearMapAO = LinearMapAA(rand(6,4) ; odim=(2,3), idim=(1,4)),
)
    (M,N) = size(O)

    @testset "O*O" begin
        @test O'*O isa LinearMapAO
    end

    @testset "O*M" begin
        B = LinearMapAA(rand(N,3)) # AM
        O1 = redim(O ; idim=(N,))
        @test O1*B isa LinearMapAO
    end

    @testset "M*O" begin
        B = LinearMapAA(rand(3,M)) # AM
        O1 = redim(O ; odim=(M,))
        @test B*O1 isa LinearMapAO
    end

    true
end


# identity and more object multiply
function test_mul_I(A::LinearMapAX)

    # multiply with identity
    @testset "*I" begin
        @test Matrix(A * 6I) == 6 * Matrix(A)
        @test Matrix(7I * A) == 7 * Matrix(A)
        @test Matrix((false*I) * A) == zeros(size(A))
        @test Matrix(A * (false*I)) == zeros(size(A))
        @test 1.0I * A === A
        @test A * 1.0I === A
        @test I * A === A
        @test A * I === A
    end

    # multiply
    @testset "*" begin
        Lm = Matrix(A)
        D = A * A'
        @test D isa LinearMapAX
        @test Matrix(D) == Lm * Lm'
        @test issymmetric(D) == true

        if (A isa LinearMapAM)
            E = A * Lm'
            @test E isa LinearMapAX
            @test Matrix(E) == Lm * Lm'
            F = Lm' * A
            @test F isa LinearMapAX
            @test Matrix(F) == Lm' * Lm
        #    @test LinearMapAA_test_getindex(F) # todo: cut if coverage is ok
        end

        @test LinearMapAA_test_mul()
    end
end


function make_L( ;
    Ta::DataType = ComplexF32, # complex to stress test
    N::Int = 6,
    M::Int = N+2, # non-square to stress test
)
    va = randn(Ta, N)
    forw = x -> [cumsum(x); va'*x; 0]
    back = y -> reverse(cumsum(reverse(y[1:N]))) + y[N+1]*va

    L = LinearMap{Ta}(forw, back, M, N)
    return M, N, L
end

M, N, L = make_L()
A = LinearMapAA(L ; T=Ta)
Lm = Matrix(L)

O = LinearMapAA(L ; odim=(1,size(A,1)), idim=(size(A,2),1))
O11 = LinearMapAA(L ; operator = true)
O1N = LinearMapAA(L ; idim=(3,2))

# check forw/back
@test Matrix(L') == Matrix(L)'
@test Matrix(A') == Matrix(A)'
@test Matrix(O') == Matrix(O)'

x = randn(N)
y = randn(M)

b = y' * O11
mul!(b, y', O11)

y' * O1N # todo: this needs more thought

throw(5)

#@test LinearMapAA_test_vmul(O) # todo tmp

# todo tmp
# AO complex case
@testset "vmul AO ℂ" begin
    B = randn(ComplexF32, 10, 12)
    O11 = LinearMapAA(B ; operator = true)
    @test LinearMapAA_test_vmul(O11)
#throw(0) # todo
    O1 = LinearMapAA(B ; odim=(10,), idim=(3,4))
    @test LinearMapAA_test_vmul(O1)
    Oc = LinearMapAA(B ; odim=(2,5), idim=(3,4))
    @test LinearMapAA_test_vmul(Oc)
end


@testset "vmul" begin
    @test LinearMapAA_test_vmul(A)
    @test LinearMapAA_test_vmul(A*A'*A) # CompositeMap
    @test LinearMapAA_test_vmul(O) # AO type
    @test ndims(O) == 2
end


# left *
@testset "left * AM" begin
    x = rand(N)
    y = rand(M)
    @test A * x ≈ Lm * x
    @test A' * y ≈ Lm' * y
    @test y' * A ≈ y' * Lm
    @test (y' * A) * x ≈ y' * (A * x) ≈ y' * Lm * x
end


# AM complex case
@testset "vmul AM ℂ" begin
    Ac = LinearMapAA(randn(ComplexF32, 5,4))
    @test LinearMapAA_test_vmul(Ac)
end


# AO complex case
@testset "vmul AO ℂ" begin
    B = randn(ComplexF32, 10, 12)
    Oc = LinearMapAA(B ; odim=(2,5), idim=(3,4))
    @test LinearMapAA_test_vmul(Oc)
end

# AO FunctionMap complex
@testset "AO FM ℂ" begin
    T = ComplexF16
    c = T(2im)
    forw! = (y,x) -> copyto!(y,x) .*= c # c*I
    back! = (x,y) -> copyto!(x,y) .*= conj(c)
    idim = (2,3)
    odim = idim # must match due to copyto!
    O = LinearMapAA(forw!, back!, (prod(odim),prod(idim)) ;
        T=T, idim=idim, odim=odim)
    x = rand(T, idim)
    @test O*x == c*x
    y = rand(T, odim)
    @test O'*y == conj(c)*y
    @test Matrix(O') == Matrix(O)'
    # todo x'O
end

test_mul_I(A)
test_mul_I(O)

@testset "AO for 1D" begin
end
    O = LinearMapAO(A)
    @test O isa LinearMapAO
    X = rand(N,2)
    Y = O * X
    @test Y isa AbstractArray
    @test Y ≈ Lm * X
    Z = O' * Y
    @test Z isa AbstractArray
    @test Z ≈ Lm' * Y
    x = rand(N)
    y = rand(M)
    @test O * x ≈ Lm * x
    @test O' * y ≈ Lm' * y
#    y' * O # todo
