# LinearMapsAA.jl

[![Build Status](https://travis-ci.org/JeffFessler/LinearMapsAA.jl.svg?branch=master)](https://travis-ci.org/JeffFessler/LinearMapsAA.jl)
[![codecov.io](http://codecov.io/github/JeffFessler/LinearMapsAA.jl/coverage.svg?branch=master)](http://codecov.io/github/JeffFessler/LinearMapsAA.jl?branch=master)
https://github.com/JeffFessler/LinearMapsAA.jl

UNDER CONSTRUCTION!

This package is an overlay for the
[`LinearMaps.jl`](https://github.com/Jutho/LinearMaps.jl)
package
that allows one to represent linear operations
(like the FFT)
as a object that appears to the user like a matrix
but internally uses fast computations
for operations, especially multiplication.

The extra `AA` in the package name has two meanings.

- Objects of type `LinearMapAA` are subtypes of `AbstractArray{T,2}`, i.e.,
[conform to the requirements of an `AbstractMatrix`](https://docs.julialang.org/en/latest/manual/interfaces/#man-interface-array-1)

- The package was developed in Ann Arbor, Michigan :)

Any `AbstractArray`
must support a `getindex` operation.
The maintainers of the `LinearMaps.jl` package
[do not wish to add getindex there](https://github.com/Jutho/LinearMaps.jl/issues/38)
so this package adds that feature
(without committing "type piracy").

A bonus feature supported by `LinearMapsAA`
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
The properties are lost when one combines
two `LinearMapAA` objects by adding, multiplying, concatenating, etc.


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
For more details see [example/fft.jl]
```
using FFTW
N = 8
A = LinearMapAA(fft, y -> N*ifft(y), (N, N), (name="fft",), T=ComplexF32)
@show A[:,2]
```

## Caution

An `AbstractArray` also must support a `setindex!` operation
and this package provides that capability,
mainly for completeness
and as a proof of principle.
A single `setindex!` call is reasonably fast,
but multiple calls add layers of complexity
that are likely to quickly slow things down.
In particular, trying to do something like the Gram-Schmidt procedure
"in place" with an `AbstractArray` would be insane.
In fact, `LinearAlgebra.qr!` only works with `StridedMatrix`
not a general `AbstractMatrix`.

## Credits

This software was developed at the
[University of Michigan](https://umich.edu/)
by
[Jeff Fessler](http://web.eecs.umich.edu/~fessler)
and his
[group](http://web.eecs.umich.edu/~fessler/group).


todo:
This package is included in the
Michigan Image Reconstruction Toolbox (MIRT.jl)
and is exported there
so that MIRT users can use it
without "separate" installation.

Being a subtype of `AbstractArray` can be useful
for other purposes,
such as using the nice
[Kronecker.jl](https://github.com/MichielStock/Kronecker.jl)
package.

For detailed installation instructions, see:
[doc/start.md](https://github.com/JeffFessler/MIRT.jl/blob/master/doc/start.md)

This package is registered in the
[`General`](https://github.com/JuliaRegistries/General) registry,
so you can install at the REPL with `] add LinearMapAA`.
