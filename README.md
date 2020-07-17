# LinearMapsAA.jl

https://github.com/JeffFessler/LinearMapsAA.jl

[![Build Status](https://travis-ci.org/JeffFessler/LinearMapsAA.jl.svg?branch=master)](https://travis-ci.org/JeffFessler/LinearMapsAA.jl)
[![codecov.io](http://codecov.io/github/JeffFessler/LinearMapsAA.jl/coverage.svg?branch=master)](http://codecov.io/github/JeffFessler/LinearMapsAA.jl?branch=master)

This package is an overlay for the package
[`LinearMaps.jl`](https://github.com/Jutho/LinearMaps.jl)
that allows one to represent linear operations
(like the FFT)
as a object that appears to the user like a matrix
but internally uses user-defined fast computations
for operations, especially multiplication.
With this package,
you can write and debug code
(especially for iterative algorithms)
using a small matrix `A`,
and then later replace it with a `LinearMapAX` object.

The extra `AA` in the package name here has two meanings.

- `LinearMapAM` is a subtype of `AbstractArray{T,2}`, i.e.,
[conforms to the requirements of an `AbstractMatrix`](https://docs.julialang.org/en/latest/manual/interfaces/#man-interface-array-1)
type.

- The package was developed in Ann Arbor, Michigan :)

An `AbstractArray`
must support a `getindex` operation.
The maintainers of the `LinearMaps.jl` package
[have not wished to add getindex there](https://github.com/Jutho/LinearMaps.jl/issues/38),
so this package adds that feature
(without committing "type piracy").

As of `v0.6`,
the package produces objects of two types:
* `LinearMapAM` (think "Matrix") that is a subtype of `AbstractMatrix`
* `LinearMapAO` (think "Operator") that is not a subtype of `AbstractMatrix`
* The general type `LinearMapAX` is a `Union` of both.
* To convert a `LinearMapAM` to a `LinearMapAO`,
use `redim` or `LinearMapAO(A)`
* To convert a `LinearMapAO` to a `LinearMapAM`, use `undim`.



## Examples

```
N = 6
L = LinearMap(cumsum, y -> reverse(cumsum(reverse(y))), N)
A = LinearMapAA(L) # version with no properties
A = LinearMapAA(L, (name="cumsum",))) # version with a NamedTuple of properties

Matrix(L), Matrix(A) # both the same 6 x 6 lower triangular matrix
A.name # returns "cumsum" here
```

Here is a more interesting example for signal processing.
```
using FFTW
N = 8
A = LinearMapAA(fft, y -> N*ifft(y), (N, N), (name="fft",), T=ComplexF32)
@show A[:,2]
```
For more details see
[example/fft.jl](https://github.com/JeffFessler/LinearMapsAA.jl/blob/master/example/fft.jl)


## Features shared with `LinearMap` objects

#### Object combinations

A `LinearMapAX` object supports all of the features of a `LinearMap`.
In particular, if `A` and `B` are both `LinearMapAX` objects
of appropriate sizes,
then the following each make new `LinearMapAX` objects:
- Multiplication: `A * B`
- Linear combination: `A + B`, `A - B`, `3A - 7B`,
- Kronecker products: `kron(A, B)`

- Concatenation: `[A B]` `[A; B]` `[I A I]` `[A B; 2A 3I]` etc.

Caution: currently some shorthand concatenations are unsupported,
like `[I I A]`, though one can accomplish that one using
`lmaa_hcat(I, I, A)`


#### Conversions

Conversion to other data types
(may require lots of memory if `A` is big):
- Convert to sparse: `sparse(A)`
- Convert to dense matrix: `Matrix(A)`


#### Avoiding memory allocations

Like `LinearMap` objects,
both types of `LinearMapAX` objects support `mul!`
for storing the results in a previously allocated output array,
to avoid new memory allocations.
The basic syntax is to replace
`y = A * x` with
`mul!(y, A, x)`.
To make the code look more like the math,
use the `InplaceOps` package:
```
using InplaceOps
@! y = A * x
```


## Features unique to `LinearMapsAA`

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

The following features are provided
by a `LinearMapAX` object
due to its `getindex` support:
- Columns or rows slicing: `A[:,5]`, `A[end,:]`etc. return a 1D vector
- Elements: `A[4,5]` (returns a scalar)
- Portions: `A[4:6,5:8]` (returns a dense matrix)
- Linear indexing: `A[2:9]` (returns a 1D vector)
- Convert to matrix: `A[:,:]` (if memory permits)
- Convert to vector: `A[:]` (if memory permits)


## Operator support

A `LinearMapAO` object
represents a linear mapping
from some input array size
into some output array size
specified by the `idim` and `odim` options.
Here is a (simplified) example for 2D MRI,
where the operator maps a 2D input array
into a 1D output vector:
```
using FFTW: fft, bfft
using LinearMapsAA
embed = (v, samp) -> setindex!(fill(zero(eltype(v)),size(samp)), v, samp)
N = (128,64) # image size
samp = rand(N...) .< 0.8 # random sampling pattern
K = sum(samp) # number of k-space samples
A = LinearMapAA(x -> fft(x)[samp], y -> bfft(embed(y,samp)),
    (K, prod(N)) ; prop = (name="fft",), T=ComplexF32, idim=N, odim=(K,))
x = rand(N...)
z = A' * (A * x) # result is a 2D array!
```
For more details see
[example/fft.jl](https://github.com/JeffFessler/LinearMapsAA.jl/blob/master/example/fft.jl)

The adjoint of this `LinearMapAO` object
maps a 1D vector of k-space samples
into a 2D image array.

Multiplying a `M × N` matrix times a `N × K` matrix
can be thought of as multiplying the matrix
by each of the `K` columns,
yielding a `M × K` result.
Generalizing this to higher dimensional arrays,
if `A::LinearMapAO`
has "input dimensions" `idim=(2,3)`
and "output dimensions" `odim=(4,5,6)`
and you do `A*X` where `X::AbstractArray` has dimension `(2,3,7,8)`,
then the output will be an `Array` of dimension `(4,5,6,7,8)`.
In other words, it works block-wise.
(If you really want a new `LinearMapAO`, rather than an `Array`,
then you must first wrap `X` in a `LinearMapAO`.)
This behavior deliberately departs from the non-`Matrix` like behavior
in `LinearMaps` where `A*X` produces a new `LinearMap`.

In the spirit of such generality,
this package overloads `*` for `LinearAlgebra.I`
(and for `UniformScaling` objects more generally)
such that
`I * X == X`
even when `X` is an array of more than two dimensions.
(The original `LinearAlgebra.I` can only multiply
vectors and matrices,
which sufficies for matrix algebra,
but not for general linear algebra.)

Caution:
The `LinearMapAM` type should be quite stable now,
whereas `LinearMapAO` is new in `v0.6`.
The conversions `redim` and `undim`
are probably not thoroughly tested.
The safe bet is to use all
`LinearMapAM` objects
or all
`LinearMapAO` objects
rather than trying to mix and match.


## Caution

An `AbstractArray` also must support a `setindex!` operation
and this package provides that capability,
mainly for completeness
and as a proof of principle,
solely for the `LinearMapAM` type.
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
a `LinearMapAM` object is designed to behave
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
In particular,
the `odim` and `idim` features of those objects
are similar to those here.

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
Michigan Image Reconstruction Toolbox
[`MIRT.jl`](https://github.com/JeffFessler/MIRT.jl)
and is exported there
so that MIRT users can use it
without "separate" installation.

Being a sub-type of `AbstractArray` can be useful
for other purposes,
such as using the nice
[Kronecker.jl](https://github.com/MichielStock/Kronecker.jl)
package.


## Compatability

* Version 0.2.0 tested with Julia 1.1 and 1.2
* Version 0.3.0 requires Julia 1.3
* Version 0.6.0 assumes Julia 1.4


## Getting started

For detailed installation instructions, see:
[doc/start.md](https://github.com/JeffFessler/MIRT.jl/blob/master/doc/start.md)

This package is registered in the
[`General`](https://github.com/JuliaRegistries/General) registry,
so you can install it at the REPL with `] add LinearMapAA`.
