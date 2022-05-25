# cuda.jl
# Test that CUDA arrays are supported

using LinearMaps: LinearMap
using LinearMapsAA: LinearMapAA
using CUDA: CuArray
import CUDA
using Test: @test

if CUDA.functional()
    isinteractive() && @info "testing CUDA"
    CUDA.allowscalar(false)

    nx, ny, nz = 5, 7, 6
    idim = (nx, ny, nz)
    odim = (nx, ny, nz, 2)

    x = rand(Float32, idim)

    forw1 = x -> cat(x, x, dims=4)
    back1 = y -> y[:,:,:,1] + y[:,:,:,2]
    forwv = x -> vec(forw1(reshape(x,idim)))
    backv = y -> vec(back1(reshape(y,odim)))

    O = LinearMapAA(forw1, back1, (prod(odim), prod(idim)); odim, idim)
    L = LinearMap(forwv, backv, prod(odim), prod(idim))
    A = LinearMapAA(forwv, backv, (prod(odim), prod(idim)))

    # check adjoint
    @test Matrix(O') == Matrix(O)'
    @test Matrix(A') == Matrix(A)'
    @test Matrix(L') == Matrix(L)'

    # check forward
    @test L * vec(x) == vec(O * x)
    @test A * vec(x) == vec(O * x)

    # check back
    y = O * x
    @test L' * vec(y) ≈ vec(O' * y)
    @test A' * vec(y) ≈ vec(O' * y)

    xg = CuArray(x)
    yo = O * xg
    yl = L * vec(xg)
    ya = A * vec(xg)
    @test yl == vec(yo)
    @test ya == vec(yo)

    xo = O' * yo
    xl = L' * vec(yo)
    xa = A' * vec(yo)
    @test xl == vec(xo)
    @test xa == vec(xo)
end
