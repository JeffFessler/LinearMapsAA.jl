# LinearMapsAA.jl

https://github.com/JeffFessler/LinearMapsAA.jl

[![action status][action-img]][action-url]
[![build status][build-img]][build-url]
[![pkgeval status][pkgeval-img]][pkgeval-url]
[![codecov.io][codecov-img]][codecov-url]
[![license][license-img]][license-url]
[![docs stable][docs-stable-img]][docs-stable-url]
[![docs dev][docs-dev-img]][docs-dev-url]

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
[conforms to (some of) the requirements of an `AbstractMatrix`](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array)
type.

- The package was developed in Ann Arbor, Michigan :)

As of `v0.6`,
the package produces objects of two types:
* `LinearMapAM` (think "Matrix") that is a subtype of `AbstractMatrix`.
* `LinearMapAO` (think "Operator") that is not a subtype of `AbstractMatrix`.
* The general type `LinearMapAX` is a `Union` of both.
* To convert a `LinearMapAM` to a `LinearMapAO`,
  use `redim` or `LinearMapAO(A)`
* To convert a `LinearMapAO` to a `LinearMapAM`, use `undim`.



## Examples

```julia
N = 6
L = LinearMap(cumsum, y -> reverse(cumsum(reverse(y))), N)
A = LinearMapAA(L) # version with no properties
A = LinearMapAA(L, (name="cumsum",))) # version with a NamedTuple of properties

Matrix(L), Matrix(A) # both the same 6 x 6 lower triangular matrix
A.name # returns "cumsum" here
```

Here is a more interesting example for signal processing.
```julia
using LinearMapsAA
using FFTW: fft, bfft
N = 8
A = LinearMapAA(fft, bfft, (N, N), (name="fft",), T=ComplexF32)
@show A[:,2]
```
For more details see the examples
in the
[documentation](https://jefffessler.github.io/LinearMapsAA.jl/dev/).


## Features shared with `LinearMap` objects

#### Object combinations

A `LinearMapAX` object supports all of the features of a `LinearMap`.
In particular, if `A` and `B` are both `LinearMapAX` objects
of appropriate sizes,
then the following each make new `LinearMapAX` objects:
- Multiplication: `A * B`
- Linear combination: `A + B`, `A - B`, `3A - 7B`
- Kronecker products: `kron(A, B)`

- Concatenation: `[A B]` `[A; B]` `[I A I]` `[A B; 2A 3I]` etc.

Caution: currently some shorthand concatenations are unsupported,
like `[I I A]`, though one can accomplish that one using
`lmaa_hcat(I, I, A)`


#### Conversions

Conversion to other data types
(may require lots of memory if `A` is big):
- Convert to sparse: `sparse(A)`
- Convert to dense matrix: `Matrix(A)`.


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
```julia
using InplaceOps
@! y = A * x # shorthand for mul!(y, A, x)
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
- Convert to vector: `A[:]` (if memory permits).


## Operator support

A `LinearMapAO` object
represents a linear mapping
from some input array size
into some output array size
specified by the `idim` and `odim` options.
Here is a (simplified) example for 2D MRI,
where the operator maps a 2D input array
into a 1D output vector:
```julia
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
typeof(A) # LinearMapAO{ComplexF32, 1, 2}
```
For more details see FFT example in the
[documentation](https://jefffessler.github.io/LinearMapsAA.jl/dev/).

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

Here is a corresponding example (not useful; just for illustration).
```julia
using LinearMapsAA
idim = (2,3)
odim = (4,5,6)
forward = x -> repeat(reshape(x, (idim[1],1,idim[2])) ; outer=(2,5,2))
A = LinearMapAA(forward,
    (prod(odim), prod(idim)) ; prop = (name="test",), idim, odim)
x = rand(idim..., 7, 8)
y = A * x
```

In the spirit of such generality,
this package overloads `*` for `LinearAlgebra.I`
(and for `UniformScaling` objects more generally)
such that
`I * X == X`
even when `X` is an array of more than two dimensions.
(The original `LinearAlgebra.I` can only multiply
vectors and matrices,
which suffices for matrix algebra,
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


## Historical note about `getindex`

An `AbstractArray`
must support a `getindex` operation.
The maintainers of the `LinearMaps.jl` package
[originally did not wish to add `getindex` there](https://github.com/Jutho/LinearMaps.jl/issues/38),
so this package added that feature
(while avoiding "type piracy").
Eventually,
partial `getindex` support,
[specifically slicing](https://github.com/JuliaLinearAlgebra/LinearMaps.jl/pull/165),
was added in
[v3.7](https://github.com/JuliaLinearAlgebra/LinearMaps.jl/releases/tag/v3.7.0)
there.
As of v0.11,
this package uses that `getindex` implementation
and also supports only slicing.
This is a breaking change that could be easily reverted,
so please submit an issue if you have a use case
for more general use of `getindex`.


## Historical note about `setindex!`

The
[Julia manual section on the `AbstractArray` interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array)
implies that an `AbstractArray`
should support a `setindex!` operation.
Versions of this package prior to v0.8.0
provided that capability,
mainly for completeness
and as a proof of principle,
solely for the `LinearMapAM` type.
However,
the reality is that many sub-types of `AbstractArray`
in the Julia ecosystem,
such as `LinearAlgebra.Diagonal`,
understandably do *not* support `setindex!`,
and it there seems to be no use
for it here either.
Supporting `setindex!` seems impossible with a concrete type
for a function map,
so it is no longer supported.
The key code is relegated to the `archive` directory.


## Related packages

[`LinearOperators.jl`](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl)
also provides `getindex`-like features,
but slicing there always returns another operator,
unlike with a matrix.
In contrast,
a `LinearMapAM` object is designed to behave
akin to a matrix,
except for operations like `svd` and `pinv`
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

[`LazyAlgebra.jl`](https://github.com/emmt/LazyAlgebra.jl)
also has many related features, and supports nonlinear mappings.

This package provides similar functionality
as the `Fatrix` / `fatrix` object in the
[Matlab version of MIRT](https://github.com/JeffFessler/mirt).
In particular,
the `odim` and `idim` features of those objects
are similar to those here.

[`FunctionOperators.jl`](https://github.com/hakkelt/FunctionOperators.jl)
supports `inDims` and `outDims` features.

Being a sub-type of `AbstractArray` can be useful
for other purposes,
such as using the nice
[Kronecker.jl](https://github.com/MichielStock/Kronecker.jl)
package.


## Inter-operability

To promote inter-operability of different linear mapping packages,
this package provides methods
for wrapping other operator types into `LinearMapAX` types.
The syntax is simply
`LinearMapAA(L; kwargs...)`
where `L` can be any of the following types currently:
* `AbstractMatrix` (including `Matrix`, `SparseMatrixCSC`, `Diagonal`, among many others)
* `LinearMap` from
  [`LinearMaps.jl`](https://github.com/Jutho/LinearMaps.jl)
* `LinearOperator` from
  [`LinearOperators.jl`](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).

Submit an issue or make a PR if there are other operator types
that one would like to have supported.
To minimize package dependencies,
the wrapping code for a `LinearOperator` uses
[Requires.jl](https://github.com/JuliaPackaging/Requires.jl).


## Multiplication properties

It can help developers and users
to know how multiplication operations should behave.

| Type | Shorthand |
| :--- | :---: |
| `LinearMapAO` | `O` |
| `LinearMapAM` | `M` |
| `LinearMap` | `L` |
| `AbstractVector` | `v` |
| `AbstractMatrix` | `X` |
| `AbstractArray` | `A` |
| `LinearAlgebra.I` | `I` |

For `left * right` multiplication the results are as follows.

| Left | Right | Result |
| :---: | :---: | :---: |
| `M` | `v` | `v` |
| `v'` | `M` | `v'` |
| `M` | `X` | `X` |
| `X` | `M` | `X` |
| `M` | `M` | `M` |
| `M` | `L` | `M` |
| `L` | `M` | `M` |
| `O` | `A` | `A` |
| `A` | `O` | `A` |
| `O` | `O` | `O` |
| `I` | `A` | `A` |

The following subset of the above operations also work
for the in-place version `mul!(result, left, right)`:

| Left | Right | Result |
| :---: | :---: | :---: |
| `M` | `v` | `v` |
| `v'` | `M` | `v'` |
| `M` | `X` | `X` |
| `X` | `M` | `X` |
| `O` | `A` | `A` |
| `A` | `O` | `A` |


There is one more special multiplication property.
If `O` is a `LinearMapAO` and `xv` is `Vector` of `AbstractArrays`,
then `O * xv` is equivalent to `[O * x for x in xv]`.
This is useful, for example,
in dynamic imaging
where one might store a video sequence
as a vector of 2D images,
rather than as a 3D array.
There is also a corresponding
[5-argument `mul!`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.mul!).
Each array in the `Vector` `xv`
must be compatible with multiplication on the left by `O`.


## Credits

This software was developed at the
[University of Michigan](https://umich.edu/)
by
[Jeff Fessler](http://web.eecs.umich.edu/~fessler)
and his
[group](http://web.eecs.umich.edu/~fessler/group),
with substantial inspiration drawn
from the `LinearMaps` package.



## Compatibility

* Version 0.2.0 tested with Julia 1.1 and 1.2
* Version 0.3.0 requires Julia 1.3
* Version 0.6.0 assumes Julia 1.4
* Version 0.7.0 assumes Julia 1.6


## Getting started

This package is registered in the
[`General`](https://github.com/JuliaRegistries/General) registry,
so you can install it at the REPL with `] add LinearMapAA`.

Here are
[detailed installation instructions](https://github.com/JeffFessler/MIRT.jl/blob/main/doc/start.md).

This package is included in the
Michigan Image Reconstruction Toolbox
[`MIRT.jl`](https://github.com/JeffFessler/MIRT.jl)
and is exported there
so that MIRT users can use it
without "separate" installation.


<!-- URLs -->
[action-img]: https://github.com/JeffFessler/LinearMapsAA.jl/workflows/Unit%20test/badge.svg
[action-url]: https://github.com/JeffFessler/LinearMapsAA.jl/actions
[build-img]: https://github.com/JeffFessler/LinearMapsAA.jl/workflows/CI/badge.svg?branch=main
[build-url]: https://github.com/JeffFessler/LinearMapsAA.jl/actions?query=workflow%3ACI+branch%3Amain
[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/L/LinearMapsAA.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/L/LinearMapsAA.html
[codecov-img]: https://codecov.io/github/JeffFessler/LinearMapsAA.jl/coverage.svg?branch=main
[codecov-url]: https://codecov.io/github/JeffFessler/LinearMapsAA.jl?branch=main
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JeffFessler.github.io/LinearMapsAA.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://JeffFessler.github.io/LinearMapsAA.jl/dev
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat
[license-url]: LICENSE
<!--
[![coveralls][coveralls-img]][coveralls-url]
[coveralls-img]: https://coveralls.io/repos/JeffFessler/LinearMapsAA.jl/badge.svg?branch
[coveralls-url]: https://coveralls.io/github/JeffFessler/LinearMapsAA.jl?branch=main
-->
