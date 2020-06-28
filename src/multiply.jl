#=
multiply.jl
Multiplication of a LinearMapAX object with other things
2020-06-16 add 5-arg mul!
2020-06-27 Jeff Fessler
=#

export mul!

# export LinearMapAA_test_vmul # testing
# export LinearMapAA_test_mul # testing

using LinearMaps
using LinearAlgebra: UniformScaling, I
import LinearAlgebra: mul! #, lmul!, rmul!
#import LinearAlgebra # AdjointAbsVec
using Test: @test, @testset



# multiply with I or s*I (identity or scaled identity)
Base.:(*)(A::LinearMapAX, B::UniformScaling{Bool}) = B.λ ? A : (A * B.λ)
Base.:(*)(A::LinearMapAX, B::UniformScaling) = (B.λ == 1) ? A : (A * B.λ)
Base.:(*)(B::UniformScaling{Bool}, A::LinearMapAX) = B.λ ? A : (B.λ * A)
Base.:(*)(B::UniformScaling, A::LinearMapAX) = (B.λ == 1) ? A : (B.λ * A)


# multiply with scalars
Base.:(*)(s::Number, A::LinearMapAX) =
    LinearMapAA((s*I) * A._lmap ; prop=A._prop, idim=A._idim, odim=A._odim)
Base.:(*)(A::LinearMapAX, s::Number) =
    LinearMapAA(A._lmap * (s*I) ; prop=A._prop, idim=A._idim, odim=A._odim)


# multiply LMAX objects, if compatible
Base.:(*)(A::LinearMapAO{Ta,Do,D}, B::LinearMapAO{Tb,D,Di}) where {Ta,Tb,Do,D,Di} =
    lm_obj_mul(A, B)
Base.:(*)(A::LinearMapAO{T,Do,1}, B::LinearMapAM) where {T,Do} = lm_obj_mul(A, B)
Base.:(*)(A::LinearMapAM, B::LinearMapAO{T,1,Di}) where {T,Di} = lm_obj_mul(A, B)
Base.:(*)(A::LinearMapAM, B::LinearMapAM) = lm_obj_mul(A, B)

function lm_obj_mul(A::LinearMapAX, B::LinearMapAX)
    (A._idim != B._odim) && throw("dim mismatch")
    LinearMapAA(A._lmap * B._lmap ;
        prop = (prod=(A._prop,B._prop),),
        idim = B._idim, odim = A._odim)
end


# multiply with a matrix
# subtle: AM * Matrix and Matrix * AM yield a new AM
# whereas AO * M and M * AO yield a matrix of numbers!
function Base.:(*)(A::LinearMapAM, B::AbstractMatrix)
    (A._idim == (size(B,1),)) || throw("$(A._idim) * $(size(B,1)) mismatch")
    LinearMapAA(A._lmap * LinearMap(B) ; prop=A._prop, odim=A._odim)
end
function Base.:(*)(A::AbstractMatrix, B::LinearMapAM)
    (B._odim == (size(A,2),)) || throw("$(B._odim) * $(size(A,1)) mismatch")
    LinearMapAA(LinearMap(A) * B._lmap ; prop=B._prop, idim=B._idim)
end


# LMAM case is easy!

# multiply with vectors (in-place)
# pass to lmam_mul! to handle composite maps (products) effectively
# 5-arg mul! requires julia 1.3 or later
# mul!(y, A, x, α, β) ≡ y .= A*(α*x) + β*y

mul!(y::AbstractVector, A::LinearMapAM, x::AbstractVector) =
    lm_mul!(y, A._lmap, x, 1, 0)
#    mul!(y, A._lmap, x)

mul!(y::AbstractVector, A::LinearMapAM, x::AbstractVector, α::Number, β::Number) =
    lm_mul!(y, A._lmap, x, α, β)
#    mul!(y, A._lmap, x, α, β)

# treat LinearMaps.CompositeMap as special case for in-place operations
function lm_mul!(y::AbstractVector, Lm::LinearMaps.CompositeMap,
    x::AbstractVector, α::Number, β::Number)
    LinearMaps.mul!(y, Lm, x, α, β) # todo: composite buffer
end

# 5-arg mul! for any other type
lm_mul!(y::AbstractVector, Lm::LinearMap,
    x::AbstractVector, α::Number, β::Number) =
    LinearMaps.mul!(y, Lm, x, α, β)

# with array
#=
these are unused because AM * array becomes a new AM
mul!(Y::AbstractArray, A::LinearMapAM, X::AbstractArray, α::Number, β::Number) =
    lmao_mul!(Y, A._lmap, X, α, β ; idim=A._idim, odim=A._odim)

mul!(Y::AbstractArray, A::LinearMapAM, X::AbstractArray) =
    LinearMapsAA.mul!(Y, A, X, 1, 0)
=#


# LMAO case


# 3-arg O*X
mul!(Y::AbstractArray, A::LinearMapAO, X::AbstractArray) =
    mul!(Y, A, X, 1, 0)

# 3-arg X*O
mul!(Y::AbstractArray, X::AbstractArray, A::LinearMapAO) =
    mul!(Y, X, A, 1, 0)

# 5-arg O*X
mul!(Y::AbstractArray, A::LinearMapAO, X::AbstractArray, α::Number, β::Number) =
    lmao_mul!(Y, A._lmap, X, α, β ; idim=A._idim, odim=A._odim)

# 5-arg X*O (todo: test complex case)
mul!(Y::AbstractArray, X::AbstractArray, A::LinearMapAO, α::Number, β::Number) =
    lmao_mul!(Y, A._lmap', X, α, β ; idim=A._odim, odim=A._idim) # note!


"""
     lmao_mul!(Y, A, X, α, β ; idim, odim)

Core routine for 5-arg multiply.
If `A._idim = (2,3,4)` and `A._odim = (5,6)` and
if input `X` has size `(2,3,4, 7,8)`
then output `Y` will have size `(5,6, 7,8)`
"""
function lmao_mul!(Y::AbstractArray, Lm::LinearMap, X::AbstractArray,
    α::Number, β::Number ;
    idim = (size(Lm,2),),
    odim = (size(Lm,1),),
)

    Di = length(idim)
    Do = length(odim)
    (Di > ndims(X) || (idim != size(X)[1:Di])) &&
        throw("idim=$(idim) vs size(RHS)=$(size(X))")
    (Do > ndims(Y) || (odim != size(Y)[1:Do])) &&
        throw("odim=$(odim) vs size(LHS)=$(size(Y))")
    size(X)[(Di+1):end] == size(Y)[(Do+1):end] ||
        throw("size(LHS)=$(size(Y)) vs size(RHS)=$(size(X))")

    x = reshape(X, prod(idim), :)
    y = reshape(Y, prod(odim), :)
    K = size(x,2)
    size(y,2) == K || throw("mismatch $(size(y,2)) K=$K")

    for k=1:K
        xk = selectdim(x,2,k)
        yk = selectdim(y,2,k)
        lm_mul!(yk, Lm, xk, α, β)
    end
    return Y
end


# multiply by array, with allocation

Base.:(*)(A::LinearMapAO, X::AbstractArray) = lmax_mul(A, X)
Base.:(*)(X::AbstractArray, A::LinearMapAO) = lmax_mul(A', X) # note!

#=
# this next line caused ambiguous method errors:
# Base.:(*)(A::LinearMapAM, X::AbstractArray) = lmax_mul(A, X)

# so i resort to this awful kludge:
Base.:(*)(A::LinearMapAM, X::AbstractArray{T,4}) where {T} = lmax_mul(A, X)

nah, too difficult, so revert to the AM*M returning an object, per above
=#

function lmax_mul(A::LinearMapAX{T}, X::AbstractArray) where {T}
    Di = length(A._idim)
    Do = length(A._odim)
    (Di > ndims(X) || (A._idim != size(X)[1:Di])) &&
         throw("idim=$(A._idim) vs size(RHS)=$(size(X))")
    extra = size(X)[(Di+1):end]
    Ty = promote_type(T, eltype(X))
    Y = Array{T}(undef, A._odim..., extra...) # allocate
    lmao_mul!(Y, A._lmap, X, 1, 0; idim=A._idim, odim=A._odim)
end


# multiply with vector

# O*v
Base.:(*)(A::LinearMapAO{T,Do,1}, v::AbstractVector) where {T,Do} =
    reshape(A._lmap * v, A._odim)
# u'*O (no, use general X*O above because unclear what this would mean)
#Base.:(*)(u::LinearAlgebra.AdjointAbsVec, A::LinearMapAO) =
#    reshape(A._lmap' * u', A._idim)

# A*v
Base.:(*)(A::LinearMapAM, v::AbstractVector) =
    A._lmap * v
# u'*A (nah, not worth it)
#Base.:(*)(u::LinearAlgebra.AdjointAbsVec, A::LinearMapAM) =
#    (A._lmap' * u')'

#= bad kludge
LMAAmanyFromOne = Union{
    LinearMapAA{T,2,1},
    LinearMapAA{T,3,1},
    LinearMapAA{T,4,1},
    } where {T}

Base.:(*)(A::LMAAmanyFromOne, v::AbstractVector{<:Number}) where {T} =
    reshape(A._lmap * v, A._odim)
Base.:(*)(A::LinearMapAA, v::AbstractVector{<:Number}) where {T} =
    reshape(A._lmap * v, A._odim)
=#


#= these are pointless; see multiplication with scalars above
lmul!(s::Number, A::LinearMapAA) = lmul!(s, A._lmap)
rmul!(A::LinearMapAA, s::Number) = rmul!(A._lmap, s)
=#


# test


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
