"""
    MultiDeviceVector{T,VP,P,GE} <: AbstractVector{T}

Dense vector distributed across CUDA devices, with each device holding a `CuVector{T}`
partition.

# Fields
- `partitions::VP` — per-device `CuVector{T}` segments
- `spec::P` — [`PartitionSpec`](@ref) describing the index distribution
- `ghost_exchange::GE` — optional [`GhostExchange`](@ref) for self-contained halo communication
  (`Nothing` when no exchange is attached)
"""
struct MultiDeviceVector{T,VP<:AbstractVector{<:CuVector{T}},P<:PartitionSpec,GE} <:
       AbstractVector{T}
    partitions::VP
    spec::P
    ghost_exchange::GE
end

function MultiDeviceVector{T}(::UndefInitializer, spec::PartitionSpec) where {T}
    partitions = Vector{CuVector{T}}(undef, spec.ndevices)
    @sync for d in 1:spec.ndevices
        @async begin
            CUDA.device!(device_id(spec, d))
            partitions[d] = CuVector{T}(undef, length(spec.ranges[d]))
        end
    end
    return MultiDeviceVector{T,Vector{CuVector{T}},typeof(spec),Nothing}(partitions, spec, nothing)
end

function MultiDeviceVector(v::Vector{T}, spec::PartitionSpec) where {T}
    @assert length(v) == spec.len "Vector length $(length(v)) != spec length $(spec.len)"
    partitions = Vector{CuVector{T}}(undef, spec.ndevices)
    @sync for d in 1:spec.ndevices
        @async begin
            CUDA.device!(device_id(spec, d))
            partitions[d] = CuVector{T}(v[spec.ranges[d]])
        end
    end
    return MultiDeviceVector{T,Vector{CuVector{T}},typeof(spec),Nothing}(partitions, spec, nothing)
end

function MultiDeviceVector(v::Vector{T}; ndevices::Int=length(CUDA.devices())) where {T}
    spec = compute_partition_ranges(length(v), ndevices)
    return MultiDeviceVector(v, spec)
end

Base.size(v::MultiDeviceVector) = (v.spec.len,)
Base.length(v::MultiDeviceVector) = v.spec.len
Base.eltype(::Type{<:MultiDeviceVector{T}}) where {T} = T

function Base.similar(v::MultiDeviceVector{T}) where {T}
    return MultiDeviceVector{T}(undef, v.spec)
end

function Base.similar(v::MultiDeviceVector, ::Type{S}) where {S}
    return MultiDeviceVector{S}(undef, v.spec)
end

function Base.similar(v::MultiDeviceVector{T}, ::Type{S}, dims::Tuple{Int}) where {T,S}
    if dims == size(v)
        return MultiDeviceVector{S}(undef, v.spec)
    end
    spec = compute_partition_ranges(dims[1], v.spec.ndevices; devices=collect(Int, v.spec.devices))
    return MultiDeviceVector{S}(undef, spec)
end

function Base.getindex(v::MultiDeviceVector, i::Int)
    @boundscheck checkbounds(v, i)
    d, li = device_for_index(v.spec, i)
    CUDA.device!(device_id(v.spec, d))
    return CUDA.@allowscalar v.partitions[d][li]
end

function Base.setindex!(v::MultiDeviceVector, val, i::Int)
    @boundscheck checkbounds(v, i)
    d, li = device_for_index(v.spec, i)
    CUDA.device!(device_id(v.spec, d))
    CUDA.@allowscalar v.partitions[d][li] = val
    return v
end

function Base.zero(v::MultiDeviceVector{T}) where {T}
    w = similar(v)
    fill!(w, zero(T))
    return w
end
