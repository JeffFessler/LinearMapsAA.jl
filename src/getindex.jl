#=
getindex.jl
Indexing support for LinearMapAX
2018-01-19, Jeff Fessler, University of Michigan
=#

Base.getindex(A::LinearMapAX, args...) = Base.getindex(A._lmap, args...)

#=

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
    return A._lmap * e
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
        return hcat([A[ii,j] for j in 1:size(A,2)]...) # very slow!
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

=#
