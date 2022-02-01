#---------------------------------------------------------
# # [Operator example: FFT](@id 03-fft)
#---------------------------------------------------------

#=
This page illustrates
the "linear operator" feature
of the Julia package
[`LinearMapsAA`](https://github.com/JeffFessler/LinearMapsAA.jl)
for the case of a multi-dimensional FFT operation.

This page was generated from a single Julia file:
[03-fft.jl](@__REPO_ROOT_URL__/03-fft.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`03-fft.ipynb`](@__NBVIEWER_ROOT_URL__/03-fft.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`03-fft.ipynb`](@__BINDER_ROOT_URL__/03-fft.ipynb).


# ### Setup

# Packages needed here.

using LinearMapsAA
using FFTW: fft, bfft, fft!, bfft!
using MIRTjim: jim, prompt
using Plots: gui
using LazyGrids: btime
using BenchmarkTools: @benchmark
using InteractiveUtils: versioninfo


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


# ### Overview

#=
A 1D N-point discrete Fourier transform (DFT)
is a linear operation
that is naturally represented as a `N × N` matrix.

The multi-dimensional DFT
is a linear mapping
that could be represented as a matrix,
using the `vec(⋅)` of its arguments,
but it is more naturally represented
as a linear operator ``A``.
For 2D images of size ``M × N``,
we can think the DFT
as an operator
``A`` that maps a ``M × N`` matrix into a ``M × N`` matrix of DFT coefficients.
In other words,
``A : \mathbb{C}^{M × N} \mapsto \mathbb{C}^{M × N}``.

The `LinearMapsAA` package
can represent such an operator easily.
=#


#=
We first define appropriate forward and adjoint functions.
We use `fft!` and `bfft!` to avoid unnecessary memory allocations.
=#

forw!(y, x) =  fft!(copyto!(y, x)) # forward mapping function
back!(x, y) = bfft!(copyto!(x, y)) # adjoint mapping function


#=
Below is the operator definition for ``(M,N) = (8,16)``.

Because FFT returns complex numbers, we must use `T=ComplexF32` here
for `LinearMaps` to work properly.
=#

M,N = 16,8
T = ComplexF32
A = LinearMapAA(forw!, back!, (M*N, M*N); idim = (M,N), odim = (M,N), T)

#=
The `idim` argument specifies
that the input is a matrix of size `M × N`
and
the `odim` argument specifies
that the output is a matrix of size `(M,N)`.
=#


#=
Here is some verification
that applying this operator
to a matrix
produces a correct result:
=#

X = ones(M,N)
@assert A * X ≈ M*N * ((1:M) .== 1)*((1:N) .== 1)' # Kronecker impulse
X = rand(T, M, N)
@assert A * X ≈ fft(X)


#=
Although
``A`` here is *not* a matrix,
we can convert it to a matrix
(at least when ``M N`` is sufficiently small)
to perhaps understand it better:
=#

Amat = Matrix(A)
using MIRTjim: jim
jim(
 jim(real(Amat)', "Real(A)"; prompt=false),
 jim(imag(Amat)', "Imag(A)"; prompt=false),
)



#=
## Adjoint

Here is a verification that the
[adjoint](https://en.wikipedia.org/wiki/Adjoint)
of the operator
``A``
is working correctly.
=#

@assert Matrix(A)' ≈ Matrix(A')


#=
Some users
might wonder if there is "overhead"
in using the overloaded linear mapping `A * x`
or `mul!(y, A, x)`
compared to directly calling
`fft!(copyto!(y), x)`.

Here are some timing tests
that confirm that `LinearMapsAA` does not incur overhead.

We deliberately choose very small `M,N`,
because any overhead will be most apparent
when the `fft` computation itself is fast.
=#


x = rand(ComplexF32, M, N)
y1 = similar(x)
y2 = similar(x)

mul!(y1, A, x)
forw!(y2, x)
@assert y1 == y2
mul!(y1, A', x)
back!(y2, x)
@assert y1 == y2

# time forward fft:
timer(t) = btime(t; scale=10^3)
t = @benchmark forw!($y2, $x)       # 19.1 us (31 alloc, 2K)
timer(t)

# compare to `LinearMapsAA` version:
t = @benchmark mul!($y1, $A, $x)    # 18.1 us (44 alloc, 4K)
timer(t)

# time backward fft:
t = @benchmark back!($y2, $x)       # 19.443 μs (31 allocations: 2.12 KiB)
timer(t)

# compare to `LinearMapsAA` version:
t = @benchmark mul!($y1, $(A'), $x) # 17.855 μs (44 allocations: 4.00 KiB)
timer(t)



# ### Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer()
versioninfo(io)
split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
