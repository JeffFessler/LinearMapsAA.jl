#=
getindex.jl
Indexing support for LinearMapAX
2018-01-19, Jeff Fessler, University of Michigan
=#


# [end]
function Base.lastindex(A::LinearMapAX)
    return prod(size(A._lmap))
end

# [?,end] and [end,?]
function Base.lastindex(A::LinearMapAX, d::Int)
    return size(A._lmap, d)
end

# A[i,j]
function Base.getindex(A::LinearMapAX, i::Int, j::Int)
    T = eltype(A)
    e = zeros(T, size(A._lmap,2)); e[j] = one(T)
    tmp = A._lmap * e
    return tmp[i]
end

# A[:,j]
# it is crucial to provide this function rather than to inherit from
# Base.getindex(A::AbstractArray, ::Colon, ::Int)
# because Base.getindex does this by iterating (I think).
function Base.getindex(A::LinearMapAX, ::Colon, j::Int)
    T = eltype(A)
    e = zeros(T, size(A,2)); e[j] = one(T)
    return A * e
end

# A[ii,j]
Base.getindex(A::LinearMapAX, ii::Indexer, j::Int) = A[:,j][ii]

# A[i,jj]
Base.getindex(A::LinearMapAX, i::Int, jj::Indexer) = A[i,:][jj]

# A[:,jj]
# this one is also important for efficiency
Base.getindex(A::LinearMapAX, ::Colon, jj::AbstractVector{Bool}) =
    A[:,findall(jj)]
Base.getindex(A::LinearMapAX, ::Colon, jj::Indexer) =
    hcat([A[:,j] for j in jj]...)

# A[ii,:]
# trick: cannot use A' for a FunctionMap with no fc
function Base.getindex(A::LinearMapAX, ii::Indexer, ::Colon)
    if (:fc in propertynames(A._lmap)) && isnothing(A._lmap.fc)
        return hcat([A[ii,j] for j in 1:size(A,2)]...)
    else
        return A'[:,ii]'
    end
end

# A[ii,jj]
Base.getindex(A::LinearMapAX, ii::Indexer, jj::Indexer) = A[:,jj][ii,:]

# A[k]
function Base.getindex(A::LinearMapAX, k::Int)
    c = CartesianIndices(size(A._lmap))[k] # is there a more elegant way?
    return A[c[1], c[2]]
end

# A[kk]
Base.getindex(A::LinearMapAX, kk::AbstractVector{Bool}) = A[findall(kk)]
Base.getindex(A::LinearMapAX, kk::Indexer) = [A[k] for k in kk]

# A[i,:]
# trick: one row slice returns a 1D ("column") vector
Base.getindex(A::LinearMapAX, i::Int, ::Colon) = A[[i],:][:]

# A[:,:] = Matrix(A)
Base.getindex(A::LinearMapAX, ::Colon, ::Colon) = Matrix(A._lmap)

# A[:]
Base.getindex(A::LinearMapAX, ::Colon) = A[:,:][:]


# test


"""
    LinearMapAA_test_getindex(A::LinearMapAX)
tests for `getindex`
"""
function LinearMapAA_test_getindex(A::LinearMapAX)
    B = Matrix(A)
    @test all(size(A) .>= (4,4)) # required by tests

    tf1 = [false; trues(size(A,1)-1)]
    tf2 = [false; trues(size(A,2)-2); false]
    ii1 = (3, 2:4, [2,4], :, tf1)
    ii2 = (2, 3:4, [1,4], :, tf2)
    for i2 in ii2
        for i1 in ii1
            @test B[i1,i2] == A[i1,i2]
        end
    end

    L = A._lmap
    test_adj = !((:fc in propertynames(L)) && isnothing(L.fc))
    if test_adj
        for i1 in ii2
            for i2 in ii1
                @test B'[i1,i2] == A'[i1,i2]
            end
        end
    end

    # "end"
    @test B[3,end-1] == A[3,end-1]
    @test B[end-2,3] == A[end-2,3]
    if test_adj
        @test B'[3,end-1] == A'[3,end-1]
    end

    # [?]
    @test B[1] == A[1]
    @test B[7] == A[7]
    if test_adj
        @test B'[3] == A'[3]
    end
    @test B[end] == A[end] # lastindex

    kk = [k in [3,5] for k = 1:length(A)] # Bool
    @test B[kk] == A[kk]

    # Some tests could rely on the fact that LinearMapAM <:i AbstractMatrix,
    # by inheriting from general Base.getindex, but all are provided here.
    @test B[[1, 3, 4]] == A[[1, 3, 4]]
    @test B[4:7] == A[4:7]

    true
end
