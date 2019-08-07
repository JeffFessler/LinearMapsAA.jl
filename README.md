# LinearMapsAA.jl

[![Build Status](https://travis-ci.org/JeffFessler/LinearMapsAA.jl.svg?branch=master)](https://travis-ci.org/JeffFessler/LinearMapsAA.jl) 
[![codecov.io](http://codecov.io/github/JeffFessler/LinearMapsAA.jl/coverage.svg?branch=master)](http://codecov.io/github/JeffFessler/LinearMapsAA.jl?branch=master) 
https://github.com/JeffFessler/LinearMapsAA.jl


This package is an overlay for the
[`LinearMaps.jl`](https://github.com/Jutho/LinearMaps.jl)
package.

The extra `AA` in the name has two meanings.
- Objects of type `LinearMapAA` are subtypes of `AbstractArray{T,2}`, i.e.,
[conform to the requirements of an `AbstractMatrix`](https://docs.julialang.org/en/latest/manual/interfaces/#man-interface-array-1)

- The package was developed in Ann Arbor, Michigan :)

Any `AbstractArray`
must support a `getindex` operation,
and the creators of the nice `LinearMaps.jl` package
[do not wish to add getindex there](https://github.com/Jutho/LinearMaps.jl/issues/38)
so I have done it here.

Another feature supported by `LinearMapsAA`
is that a user can include a `NamedTuple` of properties
with it, and then retrieve those later
using the `A.key` syntax like one would do with a struct (composite type).  
The nice folks over at `LinearMaps.jl`
[helped get me started](https://github.com/Jutho/LinearMaps.jl/issues/53)
with this feature.


## Examples

```
N = 6
A = LinearMap(cumsum, y -> reverse(cumsum(reverse(y))), N)
B = LinearMapAA(A) # version with no properties
B = LinearMapAA(A, (name="cumsum",))) # version with a NamedTuple of properties 

Matrix(B), Matrix(A) # both the same 6 x 6 lower triangular matrix
B.name # returns "cumsum" here
```

## Caution

An `AbstractArray` also must support a `setindex!` operation
and this package provides that capability,
but it probably will execute painfully slowly
in most cases.
Modifying a single "element" of a `LinearMapAA`
is probably OK,
but modifying a whole column or row
will probably be extremely slow
and is not recommended,
except for testing with very small cases.

todo:
test in-place gaussian elimination?

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
so some users

For detailed installation instructions, see:
[doc/start.md](https://github.com/JeffFessler/MIRT.jl/blob/master/doc/start.md)

todo:
This package is registered in the
[`General`](https://github.com/JuliaRegistries/General) registry,
so you can install at the REPL with `] add MIRT`.
