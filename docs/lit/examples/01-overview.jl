#=
# [LinearMapsAA overview](@id 01-overview)

This page illustrates the Julia package
[`LinearMapsAA`](https://github.com/JeffFessler/LinearMapsAA.jl).
=#

#srcURL

# ### Setup

# Packages needed here.

using LinearMapsAA
using ImagePhantoms: shepp_logan, SheppLoganToft
using MIRTjim: jim, prompt
using InteractiveUtils: versioninfo


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


#=
## Overview

Many computational imaging methods use system models
that are too large to store explicitly
as dense matrices,
but nevertheless
are represented mathematically
by a linear mapping `A`.

Often that linear map is thought of as a matrix,
but in imaging problems
it often is more convenient
to think of it as a more general linear operator.

The `LinearMapsAA` package
can represent both "matrix" versions
and "operator" versions
of linear mappings.
This page illustrates both versions
in the context of single-image
[super-resolution](https://en.wikipedia.org/wiki/Super-resolution_imaging)
imaging,
where the operator `A` maps a `M × N` image
into a coarser sampled image of size `M÷2 × N÷2`.

Here the operator `A` is akin to down-sampling,
except, rather than simple decimation,
each coarse-resolution pixel
is the average of a 2 × 2 block of pixels in the fine-resolution image.
=#

# ## System operator (linear mapping) for down-sampling

# Here is the "forward" function needed to model 2× down-sampling:

down1 = (x) -> (x[1:2:end,:] + x[2:2:end,:])/2 # 1D down-sampling by 2×
down2 = (x) -> down1(down1(x)')'; # 2D down-sampling by factor of 2×

# The `down2` function is a (bounded) linear operator
# and here is its adjoint:
down2_adj(y::AbstractMatrix{<:Number}) = kron(y, fill(0.25, (2,2)));


#=
Mathematically, and adjoint is a generalization
of the (Hermitian) transpose of a matrix.
For a (bounded) linear mapping `A` between
inner product space X
with inner product <.,.>_X
and inner product space Y
with inner product <.,.>_Y,
the adjoint of `A`, denoted `A'`,
is the unique bound linear mapping
that satisfies
<A x, y>_Y = <x, A' y>_X
for all x ∈ X and y ∈ Y.
One can verify that the `down2_adj` function
satisfies that equality
for the usual inner product
on the space of `M × N` images.
=#


#=
## LinearMap as an operator for super-resolution

We now pick a specific image size
and define the linear mapping
using the two functions above:
=#

nx, ny = 200, 256
A = LinearMapAA(down2, down2_adj, ((nx÷2)*(ny÷2), nx*ny);
    idim = (nx,ny), odim = (nx,ny) .÷ 2)

#=
The `idim` argument specifies
that the input image is of size `nx × ny`
and
the `odim` argument specifies
that the output image is of size `nx÷2 × ny÷2`.
This means that when we invoke
`y = A * x`
the input `x` must be a 2D array of size `nx × ny`
(not a 1D vector!)
and the output `y` will have size `nx÷2 × ny÷2`.
This behavior is a generalization
of what one might expect
from a conventional matrix-vector expression,
but is quite appropriate and convenient
for imaging problems.

Here is an illustration.
We start with a 2D test image.
=#

image = shepp_logan(ny, SheppLoganToft())[(ny-nx)÷2 .+ (1:nx),:]
jim(image, "SheppLogan")


# Apply the operator `A` to this image to down-sample it:

down = A * image
jim(down, title="Down-sampled image")


# Apply the adjoint of `A` to that result to "up-sample" it:
up = A' * down
jim(up, title="Adjoint: A' * y")


# That up-sampled image does not have the same range of values
# as the original image because `A'` is an adjoint, not an inverse!


#=
## AbstractMatrix version

Some users may prefer that the operator `A` behave more like a matrix.
We can implement approach from the same ingredients
by using `vec` and `reshape` judiciously.
The code is less elegant,
but similarly efficient
because `vec` and `reshape` are non-allocating operations.
=#

B = LinearMapAA(
        x -> vec(down2(reshape(x,nx,ny))),
        y -> vec(down2_adj(reshape(y,Int(nx/2),Int(ny/2)))),
        ((nx÷2)*(ny÷2), nx*ny),
    )

#=
To apply this version to our `image`
we must first vectorize it
because the expected input is a vector in this case.
And then we have to reshape the vector output
as a 2D array to look at it.
(This is why the operator version with `idim` and `odim` is preferable.)
=#

y = B * vec(image) # This would fail here without the `vec`!
jim(reshape(y, nx÷2, ny÷2)) # Annoying reshape needed here!


#=
Even though we write `y = A * x` above,
one must remember that internally `A` is not stored as a dense matrix.
It is simply a special variable type
that stores the forward function `down2` and the adjoint function `down2_adj`,
along with a few other values like `nx,ny`,
and applies those functions when needed.
Nevertheless,
we can examine elements of `A` and `B`
just like one would with any matrix,
at least for small enough examples to fit in memory.
=#


# Examine `A` and `A'`

nx, ny = 8,6
idim = (nx,ny)
odim = (nx,ny) .÷ 2
A = LinearMapAA(down2, down2_adj, ((nx÷2)*(ny÷2), nx*ny); idim, odim)

# Here is `A` shown as a Matrix:
jim(Matrix(A)', "A")

# Here is `A'` shown as a Matrix:
jim(Matrix(A')', "A'")


#=
When defining the adjoint function of a linear mapping,
it is very important to verify
that it is correct (truly the adjoint).

For a small problem we simply use the following test:
=#
@assert Matrix(A)' == Matrix(A')

# For some applications we must check approximate equality like this:
@assert Matrix(A)' ≈ Matrix(A')


# Here is a statistical test that is suitable for large operators.
# Often one would repeat this test several times:
T = eltype(A)
x = randn(T, idim)
y = randn(T, odim)

@assert sum((A*x) .* y) ≈ sum(x .* (A'*y)) # <Ax,y> = <x,A'y>


include("../../../inc/reproduce.jl")
