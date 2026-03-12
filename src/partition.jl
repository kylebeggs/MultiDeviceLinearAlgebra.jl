struct PartitionSpec
    ranges::Vector{UnitRange{Int}}
    len::Int
    ndevices::Int
end

function PartitionSpec(ranges::Vector{UnitRange{Int}})
    isempty(ranges) && throw(ArgumentError("Ranges vector must be non-empty"))
    for (i, r) in enumerate(ranges)
        isempty(r) && throw(ArgumentError("Range $i is empty: $r"))
    end
    first(ranges[1]) == 1 || throw(ArgumentError("First range must start at 1, got $(first(ranges[1]))"))
    for i in 1:(length(ranges) - 1)
        if last(ranges[i]) + 1 != first(ranges[i + 1])
            throw(ArgumentError("Ranges $i and $(i+1) are not contiguous: $(ranges[i]) and $(ranges[i+1])"))
        end
    end
    return PartitionSpec(ranges, last(ranges[end]), length(ranges))
end

function compute_partition_ranges(n::Int, ndevices::Int)
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

    return PartitionSpec(ranges, n, ndevices)
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
