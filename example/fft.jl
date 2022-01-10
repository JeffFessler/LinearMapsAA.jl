#=
example/fft.jl

1D FFT example

Because FFT returns complex numbers, we must use T=ComplexF32 here
for LinearMap to work properly.

Because the usual `fft` is not the unitary FFT, we need a factor of N
for the adjoint.
=#

using FFTW
using LinearMapsAA: LinearMapAA
using LinearMaps: LinearMap
using LinearAlgebra: mul!
using Test: @test

N = 8
A = LinearMapAA(fft, y -> N*ifft(y), (N, N), (name="fft",), T=ComplexF32)
@show A[:,2]

@test Matrix(A') == Matrix(A)' # test the adjoint


# timing tests to confirm that LinearMapAA does not have overhead

using BenchmarkTools

N = 2^11
L = LinearMap{ComplexF32}(fft, y -> N*ifft(y), N, N)
A = LinearMapAA(fft, y -> N*ifft(y), (N, N), (name="fft",), T=ComplexF32)

x = rand(N)
y = rand(Float32, N) + 1im * rand(Float32, N)
y2 = rand(Float32, N) + 1im * rand(Float32, N)

if true
    mul!(y, L, x)
    mul!(y2, A, x)
    @test isapprox(y, y2)
end

if true # essentially identical
    println("adjoint:")
    @btime x = ($L)' * $y
    @btime x = ($A)' * $y
end

if true # essentially identical
    println("forward:")
    @btime y = $L * $x
    @btime y = $A * $x
end

if true # essentially identical
    println("mul!:")
    @btime mul!($y, $L, $x)
    @btime mul!($y, $A, $x)
end

nothing
