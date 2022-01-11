#---------------------------------------------------------
# # [LinearMapsAA overview](@id 01-overview)
#---------------------------------------------------------

#=
This page illustrates the Julia package
[`LinearMapsAA`](https://github.com/JeffFessler/LinearMapsAA.jl).

This page was generated from a single Julia file:
[01-overview.jl](@__REPO_ROOT_URL__/01-overview.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`01-overview.ipynb`](@__NBVIEWER_ROOT_URL__/01-overview.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`01-overview.ipynb`](@__BINDER_ROOT_URL__/01-overview.ipynb).


# ### Setup

# Packages needed here.

using LinearMapsAA
using ImagePhantoms: shepp_logan, SheppLoganToft
using MIRTjim: jim, prompt
using InteractiveUtils: versioninfo


# The following line is helpful when running this example.jl file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


# ### Overview

#=
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
but rather than simple decimation
each coarse-resolution pixel
is the average of a 2 × 2 block of pixels in the fine-resolution image.
=#

nx, ny = 200, 256
image = shepp_logan(ny, SheppLoganToft())[(ny-nx)÷2 .+ (1:nx),:]
jim(image, "SheppLogan")


# ### Down-sampling linear mapping

down1 = (x) -> (x[1:2:end,:] + x[2:2:end,:])/2 # 1D down-sampling by 2x
down2 = (x) -> down1(down1(x)')' # 2D down-sampling by factor of 2x

# adjoint:
down2_adj(y::AbstractMatrix{<:Number}) =  kron(y, fill(0.25, (2,2)))

A = LinearMapAA(
    x -> vec(down2(reshape(x,nx,ny))),
    y -> vec(down2_adj(reshape(y,Int(nx/2),Int(ny/2)))),
    ((nx÷2)*(ny÷2), nx*ny),
    )

A = LinearMapAA(down2, down2_adj, ((nx÷2)*(ny÷2), nx*ny);
    idim = (nx,ny), odim = (nx,ny) .÷ 2)

down = A * image

jim(down, title="Down-sampled image")

up = A' * down

jim(up, title="Adjoint: A' * y")

# Examine A and A'

nx, ny = 8,6
A = LinearMapAA(down2, down2_adj, ((nx÷2)*(ny÷2), nx*ny);
    idim = (nx,ny), odim = (nx,ny) .÷ 2)

jim(Matrix(A)', "A")

jim(Matrix(A')', "A'")

@assert Matrix(A)' == Matrix(A')


# ## Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer()
versioninfo(io)
split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
