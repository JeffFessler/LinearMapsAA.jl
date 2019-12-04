# LinearMapsAA.jl

[![Build Status](https://travis-ci.org/JeffFessler/LinearMapsAA.jl.svg?branch=master)](https://travis-ci.org/JeffFessler/LinearMapsAA.jl)
[![codecov.io](http://codecov.io/github/JeffFessler/LinearMapsAA.jl/coverage.svg?branch=master)](http://codecov.io/github/JeffFessler/LinearMapsAA.jl?branch=master)
https://github.com/JeffFessler/LinearMapsAA.jl

This package is an overlay for the package
[`LinearMaps.jl`](https://github.com/Jutho/LinearMaps.jl)
that allows one to represent linear operations
(like the FFT)
as a object that appears to the user like a matrix
but internally uses user-defined fast computations
for operations, especially multiplication.

The extra `AA` in the package name here has two meanings.

- `LinearMapAA` is a subtype of `AbstractArray{T,2}`, i.e.,
[conforms to the requirements of an `AbstractMatrix`](https://docs.julialang.org/en/latest/manual/interfaces/#man-interface-array-1)
type.

- The package was developed in Ann Arbor, Michigan :)

An `AbstractArray`
must support a `getindex` operation.
The maintainers of the `LinearMaps.jl` package
[have not wished to add getindex there](https://github.com/Jutho/LinearMaps.jl/issues/38),
so this package adds that feature
(without committing "type piracy").

A bonus feature provided by `LinearMapsAA`
is that a user can include a `NamedTuple` of properties
with it, and then retrieve those later
using the `A.key` syntax like one would do with a struct (composite type).  
The nice folks over at `LinearMaps.jl`
[helped get me started](https://github.com/Jutho/LinearMaps.jl/issues/53)
with this feature.
Often linear operators are associated
with some properties,
e.g.,
a wavelet transform arises
from some mother wavelet,
and it can be convenient
to carry those properties with the object itself.
Currently
the properties are lost when one combines
two or more `LinearMapAA` objects by adding, multiplying, concatenating, etc.


## Examples

```
N = 6
L = LinearMap(cumsum, y -> reverse(cumsum(reverse(y))), N)
A = LinearMapAA(L) # version with no properties
A = LinearMapAA(L, (name="cumsum",))) # version with a NamedTuple of properties

Matrix(L), Matrix(A) # both the same 6 x 6 lower triangular matrix
A.name # returns "cumsum" here
```

Here is a more interesting example for computational imaging.
```
using FFTW
N = 8
A = LinearMapAA(fft, y -> N*ifft(y), (N, N), (name="fft",), T=ComplexF32)
@show A[:,2]
```
For more details see
[example/fft.jl](https://github.com/JeffFessler/LinearMapsAA.jl/blob/master/example/fft.jl)


## Features

A `LinearMapAA` object supports all of the features of a `LinearMap`.
In particular, if `A` and `B` are both `LinearMapAA` objects
of appropriate sizes,
then the following each make new `LinearMapAA` objects:
- Multiplication: `A * B`
- Linear combination: `A + B`, `A - B`, `3A - 7B`,

Conversion to other data types
(may require lots of memory if `A` is big):
- Convert to sparse: `sparse(A)`
- Convert to dense matrix: `Matrix(A)`

- Concatenation: `[A B]` `[A; B]` `[I A I]` `[A B; 2A 3I]` etc.

Caution: currently some shorthand concatenations are unsupported,
like `[I I A]`, though one can accomplish that one using
`lmaa_hcat(I, I, A)`

The following features are provided
by a `LinearMapAA` object
due to its `getindex` support:
- Columns or rows slicing: `A[:,5]`, `A[end,:]`etc. return a 1D vector
- Elements: `A[4,5]` (returns a scalar)
- Portions: `A[4:6,5:8]` (returns a dense matrix)
- Linear indexing: `A[2:9]` (returns a 1D vector)
- Convert to matrix: `A[:,:]` (if memory permits)
- Convert to vector: `A[:]` (if memory permits)


## Caution

An `AbstractArray` also must support a `setindex!` operation
and this package provides that capability,
mainly for completeness
and as a proof of principle.
Examples:
- `A[2,3] = 7`
- `A[:,4] = ones(size(A,1))`
- `A[end] = 0`

A single `setindex!` call is reasonably fast,
but multiple calls add layers of complexity
that are likely to slow things down.
In particular, trying to do something like the Gram-Schmidt procedure
"in place" with an `AbstractArray` would be insane.
In fact, `LinearAlgebra.qr!` works only with a `StridedMatrix`
not a general `AbstractMatrix`.

## Related packages

[`LinearOperators.jl`](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl)
also provides `getindex`-like features,
but slicing there always returns another operator,
unlike with a matrix.
In contrast,
a `LinearMapAA` object is designed to behave
akin to a matrix,
except when for operations like `svd` and `pinv`
that are unsuitable for large-scale problems.
However, one can try
[`Arpack.svds(A)`](https://julialinearalgebra.github.io/Arpack.jl/latest/index.html#Arpack.svds)
to compute a few SVD components.

[`LazyArrays.jl`](https://github.com/JuliaArrays/LazyArrays.jl)
and
[`BlockArrays.jl`](https://github.com/JuliaArrays/BlockArrays.jl)
also have some related features,
but only for arrays,
not linear operators defined by functions,
so `LinearMaps` is more comprehensive.

This package provides similar functionality
as the `Fatrix` / `fatrix` object in the
[Matlab version of MIRT](https://github.com/JeffFessler/mirt).
Currently the `odim` and `idim` features of those objects
are not available here,
but I hope to add such support.

[`FunctionOperators.jl`](https://github.com/hakkelt/FunctionOperators.jl)
supports `inDims` and `outDims` features.

## Credits

This software was developed at the
[University of Michigan](https://umich.edu/)
by
[Jeff Fessler](http://web.eecs.umich.edu/~fessler)
and his
[group](http://web.eecs.umich.edu/~fessler/group),
with substantial inspiration drawn
from the `LinearMaps` package.


This package is included in the
Michigan Image Reconstruction Toolbox (MIRT.jl)
and is exported there
so that MIRT users can use it
without "separate" installation.

Being a sub-type of `AbstractArray` can be useful
for other purposes,
such as using the nice
[Kronecker.jl](https://github.com/MichielStock/Kronecker.jl)
package.


## Compatability

Tested with Julia 1.1 and 1.2


## Getting started

For detailed installation instructions, see:
[doc/start.md](https://github.com/JeffFessler/MIRT.jl/blob/master/doc/start.md)

This package is registered in the
[`General`](https://github.com/JuliaRegistries/General) registry,
so you can install it at the REPL with `] add LinearMapAA`.
