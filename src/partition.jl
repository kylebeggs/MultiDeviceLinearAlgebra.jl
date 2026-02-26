struct PartitionSpec
    ranges::Vector{UnitRange{Int}}
    len::Int
    ndevices::Int
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
