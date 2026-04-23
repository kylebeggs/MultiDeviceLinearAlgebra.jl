"""
    PartitionSpec{R,D}

Immutable metadata describing how indices are partitioned across GPU devices.

# Fields
- `ranges::R` — index ranges assigned to each device
- `len::Int` — total number of indices across all partitions
- `ndevices::Int` — number of devices
- `devices::D` — 0-indexed CUDA device IDs for each partition
"""
struct PartitionSpec{R <: AbstractVector{UnitRange{Int}}, D <: AbstractVector{Int}}
    ranges::R
    len::Int
    ndevices::Int
    devices::D
end

"""
    device_id(spec::PartitionSpec, d::Int)

Return the 0-indexed CUDA device ID for partition `d`.
"""
device_id(spec::PartitionSpec, d::Int) = spec.devices[d]

function _validate_devices(devices::AbstractVector{Int}, ndevices::Int)
    length(devices) == ndevices || throw(
        ArgumentError("Length of devices ($(length(devices))) must equal ndevices ($ndevices)")
    )
    all(>=(0), devices) || throw(
        ArgumentError("All device IDs must be non-negative, got $devices")
    )
    return length(unique(devices)) == length(devices) || throw(
        ArgumentError("Device IDs must be unique, got $devices")
    )
end

function PartitionSpec(
        ranges::AbstractVector{<:UnitRange{<:Integer}}; devices::Union{Nothing, AbstractVector{Int}} = nothing
    )
    isempty(ranges) && throw(ArgumentError("Ranges vector must be non-empty"))

    vranges = Vector{UnitRange{Int}}(undef, length(ranges))
    for (i, r) in enumerate(ranges)
        vranges[i] = UnitRange{Int}(Int(first(r)), Int(last(r)))
    end

    for (i, r) in enumerate(vranges)
        isempty(r) && throw(ArgumentError("Range $i is empty: $r"))
    end
    first(vranges[1]) == 1 || throw(ArgumentError("First range must start at 1, got $(first(vranges[1]))"))
    for i in 1:(length(vranges) - 1)
        if last(vranges[i]) + 1 != first(vranges[i + 1])
            throw(ArgumentError("Ranges $i and $(i + 1) are not contiguous: $(vranges[i]) and $(vranges[i + 1])"))
        end
    end
    ndevices = length(vranges)
    if devices === nothing
        return PartitionSpec(vranges, last(vranges[end]), ndevices, 0:(ndevices - 1))
    else
        _validate_devices(devices, ndevices)
        return PartitionSpec(vranges, last(vranges[end]), ndevices, devices)
    end
end

function compute_partition_ranges(n::Int, ndevices::Int; devices::Union{Nothing, AbstractVector{Int}} = nothing)
    @assert n > 0 "Length must be positive, got $n"
    @assert ndevices > 0 "Number of devices must be positive, got $ndevices"
    @assert ndevices <= n "More devices ($ndevices) than elements ($n)"

    base_size = div(n, ndevices)
    remainder = mod(n, ndevices)

    ranges = Vector{UnitRange{Int}}(undef, ndevices)
    offset = 0
    for d in 1:ndevices
        chunk = base_size + (d <= remainder ? 1 : 0)
        ranges[d] = (offset + 1):(offset + chunk)
        offset += chunk
    end

    if devices === nothing
        return PartitionSpec(ranges, n, ndevices, 0:(ndevices - 1))
    else
        _validate_devices(devices, ndevices)
        return PartitionSpec(ranges, n, ndevices, devices)
    end
end

function compute_partition_ranges(n::Int; devices::AbstractVector{Int})
    return compute_partition_ranges(n, length(devices); devices = devices)
end

function device_for_index(spec::PartitionSpec, i::Int)
    @boundscheck (1 <= i <= spec.len) || throw(BoundsError(spec, i))
    for d in 1:spec.ndevices
        r = spec.ranges[d]
        if i in r
            return (d, i - first(r) + 1)
        end
    end
    error("Index $i not found in any partition (bug)")
end
