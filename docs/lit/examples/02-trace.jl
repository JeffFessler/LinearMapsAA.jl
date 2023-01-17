#=
# [Operator example: trace](@id 02-trace)

This page illustrates
the "linear operator" feature
of the Julia package
[`LinearMapsAA`](https://github.com/JeffFessler/LinearMapsAA.jl).

This page was generated from a single Julia file:
[02-trace.jl](@__REPO_ROOT_URL__/02-trace.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](https://nbviewer.org/) here:
#md # [`02-trace.ipynb`](@__NBVIEWER_ROOT_URL__/02-trace.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`02-trace.ipynb`](@__BINDER_ROOT_URL__/02-trace.ipynb).


# ### Setup

# Packages needed here.

using LinearMapsAA
using LinearAlgebra: tr, I
using InteractiveUtils: versioninfo


#=
## Overview

The "operator" aspect of this package
may seem unfamiliar
to some readers
who are used to thinking in terms of matrices and vectors,
so this page
describes a simple example:
the matrix
[trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra))
operation.
The trace of a ``N × N`` matrix
is the sum of its ``N`` diagonal elements.
We tend to think of this a function,
and indeed it is the `tr` function
in the `LinearAlgebra` package.
But it is a *linear* function
so we can represent it as a linear operator
``𝒜`` that maps a ``N × N`` matrix into its trace.
In other words,
``𝒜 : \mathbb{C}^{N × N} \mapsto \mathbb{C}``
is defined by
``𝒜 X = \mathrm{tr}(X)``.
Note that the product
``𝒜 X``
is *not* a "matrix vector" product;
it is a linear operator acting on the matrix ``X``.

(Note that we use a
fancy
[unicode character](https://en.wikipedia.org/wiki/Mathematical_Alphanumeric_Symbols#Chart_for_the_Mathematical_Alphanumeric_Symbols_block)
``𝒜`` here
just as a reminder that it is an operator;
in practical code we usually just use `A`.)

The `LinearMapsAA` package
can represent such an operator easily.
Here is the definition for ``N = 5``:
=#

N = 5
forw(X) = [tr(X)] # forward mapping function
𝒜 = LinearMapAA(forw, (1, N*N); idim = (N,N), odim = (1,))
#src B = LinearMapAA(X -> tr(X), (1, N*N); idim = (N,N), odim = ()) # fails

#=
The `idim` argument specifies
that the input is a matrix of size `N × N`
and
the `odim` argument specifies
that the output is vector of size `(1,)`.
=#

#=
One subtlety with this particular didactic example
is that the ordinary trace yields a scalar,
but
[`LinearMaps.jl`](https://github.com/JuliaLinearAlgebra/LinearMaps.jl)
is (understandably) designed exclusively
for mapping vectors to vectors,
so we use `[tr(X)]`
above so that the output is a 1-element `Vector`.
This behavior
is consistent with what happens
when one multiplies
a `1 × N` matrix with a vector in ``\mathbb{C}^N``.
=#

#=
Here is a verification
that applying this operator
to a matrix
produces the correct result:
=#

X = ones(5)*(1:5)'
𝒜 * X, tr(X), (N*(N+1))÷2

#=
Although
``𝒜`` here
is *not* a matrix,
we can convert it to a matrix
(at least when ``N`` is sufficiently small)
to perhaps understand it better:
=#

A = Matrix(𝒜)
A = Int8.(A) # just for nicer display

#=
The pattern of 0 and 1 elements
is more obvious
when reshaped:
=#

reshape(A, N, N)


#=
## Adjoint

Although this is largely a didactic example,
there are optimization problems
with
[trace constraints](https://www.optimization-online.org/DB_HTML/2018/08/6765.html)
of the form
``𝒜 X = b``.
To solve such problems,
often one would also need the
[adjoint](https://en.wikipedia.org/wiki/Adjoint)
of the operator
``𝒜``.

Mathematically, and adjoint is a generalization
of the (Hermitian) transpose of a matrix.
For a (bounded) linear mapping ``𝒜`` between
inner product space ``𝒳``
with inner product ``⟨ \cdot, \cdot \rangle_𝒳``
and inner product space ``𝒴``
with inner product ``⟨ \cdot, \cdot \rangle_𝒴,``
the adjoint of ``𝒜``, denoted ``𝒜'``,
is the unique bound linear mapping
that satisfies
``⟨ 𝒜 x, y \rangle_𝒴 = ⟨ x, 𝒜' y \rangle_𝒳``
for all ``x ∈ 𝒳`` and ``y ∈ 𝒴``.

Here, let
``𝒳`` denote the vector space of ``N × N`` matrices
with the
Frobenius inner product for matrices:
``⟨ A, B \rangle_𝒳 = \mathrm{tr}(A'B)``.
Let
``𝒴``
simply be ``\mathbb{C}^1``
with the usual inner product
``⟨ x, y \rangle_𝒴 = x_1^* y_1``.

With those definitions,
one can verify that the adjoint of
``𝒜``
is the mapping
``𝒜' c = c_1 \mathbf{I}_N``,
for ``c ∈ \mathbb{C}^1``,
where
``\mathbf{I}_N``
denotes the
``N × N`` identity matrix.

Here is the `LinearMapAO`
that includes the adjoint:
=#

back(y) = y[1] * I(N) # remember `y` is a 1-vector
𝒜 = LinearMapAA(forw, back, (1, N*N); idim = (N,N), odim = (1,))

# Here is a verification that the adjoint is correct (very important!):

@assert Matrix(𝒜)' == Matrix(𝒜')
Int8.(Matrix(𝒜'))


# ### Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer()
versioninfo(io)
split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
