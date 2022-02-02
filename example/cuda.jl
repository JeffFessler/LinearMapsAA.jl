#=
example/cuda.jl
Verify that this package works with CUDA arrays.
=#

using CUDA: cu
using FFTW: fft, bfft, fft!, bfft!
using LinearMapsAA: LinearMapAA
using LinearAlgebra: mul!

N = 8
T = ComplexF32
A = LinearMapAA(fft, bfft, (N, N), (name="fft",); T)
@assert Matrix(A') == Matrix(A)' # test the adjoint

x = randn(T, N)
y = A * x

xc = cu(x)

#=
if true # essentially identical
    @btime mul!($y, $L, $x)
    @btime mul!($y, $A, $x)
end
=#
