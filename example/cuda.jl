#=
example/cuda.jl
Verify that this package works with CUDA arrays.
(This code can run only on a system with a NVidia GPU.)
=#

using CUDA: cu
import CUDA: allowscalar
using FFTW: fft, bfft, fft!, bfft!
using LinearMapsAA: LinearMapAA
using LinearAlgebra: mul!

allowscalar(false) # ensure no scalar operations


N = 8
T = ComplexF32
A = LinearMapAA(fft, bfft, (N, N), (name="fft",); T)
@assert Matrix(A') == Matrix(A)' # test the adjoint

x = randn(T, N)
y = A * x

xc = cu(x)
yc = A * xc

@assert Array(yc) ≈ y

bc = A' * yc
b = A' * y
@assert Array(bc) ≈ b


M,N = 16,8
T = ComplexF32
forw!(y,x) =  fft!(copyto!(y,x))
back!(x,y) = bfft!(copyto!(x,y))
B = LinearMapAA(forw!, back!, (N*M, N*M), (name="fft",); idim = (M,N), odim = (M,N), T)
@assert Matrix(B') == Matrix(B)'


# test fft

x = randn(T, M, N)
y = similar(x)
mul!(y, B, x)

xc = cu(x)
yc = similar(xc)
mul!(yc, B, xc)

@assert Array(yc) ≈ y


# test bfft

b = similar(x)
mul!(b, B', y)

bc = similar(yc)
mul!(bc, B', yc)

@assert Array(bc) ≈ b
