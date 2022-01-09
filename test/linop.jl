# linop.jl

using LinearMapsAA
using LinearOperators

forw! = cumsum!
back! = (x, y) -> reverse!(cumsum!(x, reverse!(copyto!(x, y))))

N = 9
L = LinearOperator(Float32, N, N, false, false, forw!,
     nothing, # will be inferred
     back!,
)

x = rand(N)
y1 = L * x
@assert y1 ≈ cumsum(x)

A = LinearMapAA(L)
