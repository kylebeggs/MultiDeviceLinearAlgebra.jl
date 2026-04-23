"""
    _compute_ghost_topology(ghost_global_indices, spec)

Compute ghost communication topology from per-device ghost index lists.
Returns `(neighbors, send_local_indices, recv_ghost_offsets, neighbor_reverse)`.
"""
function _compute_ghost_topology(
        ghost_global_indices::Vector{Vector{Int}},
        spec::PartitionSpec,
    )
    ndevices = spec.ndevices

    # ghost_by_owner[d][owner] = sorted global indices that device d needs from owner
    ghost_by_owner = [Dict{Int, Vector{Int}}() for _ in 1:ndevices]
    for d in 1:ndevices
        for g in ghost_global_indices[d]
            owner, _ = device_for_index(spec, g)
            owner_list = get!(Vector{Int}, ghost_by_owner[d], owner)
            push!(owner_list, g)
        end
    end

    # needs_from[d][requester] = global indices that requester needs from device d
    needs_from = [Dict{Int, Vector{Int}}() for _ in 1:ndevices]
    for d in 1:ndevices
        for (owner, indices) in ghost_by_owner[d]
            needs_from[owner][d] = indices
        end
    end

    # Symmetric neighbor lists: union of recv-from and send-to devices
    neighbors = Vector{Vector{Int}}(undef, ndevices)
    for d in 1:ndevices
        recv_from = Set(keys(ghost_by_owner[d]))
        send_to = Set(keys(needs_from[d]))
        neighbors[d] = sort!(collect(union(recv_from, send_to)))
    end

    # recv_ghost_offsets[d][k] = range in ghost portion where neighbor k's values land
    recv_ghost_offsets = Vector{Vector{UnitRange{Int}}}(undef, ndevices)
    for d in 1:ndevices
        recv_ghost_offsets[d] = Vector{UnitRange{Int}}(undef, length(neighbors[d]))
        offset = 0
        for (k, nbr) in enumerate(neighbors[d])
            n_from = length(get(ghost_by_owner[d], nbr, Int[]))
            recv_ghost_offsets[d][k] = (offset + 1):(offset + n_from)
            offset += n_from
        end
    end

    # send_local_indices[d][k] = local indices on d to pack for neighbor k
    send_local_indices = Vector{Vector{Vector{Int}}}(undef, ndevices)
    for d in 1:ndevices
        owned_first = first(spec.ranges[d])
        send_local_indices[d] = Vector{Vector{Int}}(undef, length(neighbors[d]))
        for (k, nbr) in enumerate(neighbors[d])
            globals_needed = get(needs_from[d], nbr, Int[])
            send_local_indices[d][k] = [g - owned_first + 1 for g in globals_needed]
        end
    end

    # neighbor_reverse[d][k] = index of d in neighbors[neighbors[d][k]]'s neighbor list
    neighbor_reverse = Vector{Vector{Int}}(undef, ndevices)
    for d in 1:ndevices
        neighbor_reverse[d] = Vector{Int}(undef, length(neighbors[d]))
        for (k, nbr) in enumerate(neighbors[d])
            neighbor_reverse[d][k] = findfirst(==(d), neighbors[nbr])
        end
    end

    return neighbors, send_local_indices, recv_ghost_offsets, neighbor_reverse
end

"""
    _compute_ghost_map(csr_rowptr, csr_colval, row_spec)

Discover ghost indices from CSR sparsity pattern, then compute communication topology.
Returns `(ghost_global_indices, neighbors, send_local_indices, recv_ghost_offsets, neighbor_reverse)`.
"""
function _compute_ghost_map(
        csr_rowptr::AbstractVector{Ti},
        csr_colval::AbstractVector{Ti},
        row_spec::PartitionSpec,
    ) where {Ti <: Integer}
    ndevices = row_spec.ndevices
    ghost_global_indices = Vector{Vector{Int}}(undef, ndevices)

    for d in 1:ndevices
        owned_range = row_spec.ranges[d]
        ghost_set = Set{Int}()
        rp_start = csr_rowptr[first(owned_range)]
        rp_end = csr_rowptr[last(owned_range) + 1] - 1
        for idx in rp_start:rp_end
            col = Int(csr_colval[idx])
            if !(col in owned_range)
                push!(ghost_set, col)
            end
        end
        ghost_global_indices[d] = sort!(collect(ghost_set))
    end

    neighbors, send_local_indices, recv_ghost_offsets, neighbor_reverse =
        _compute_ghost_topology(ghost_global_indices, row_spec)

    return ghost_global_indices, neighbors, send_local_indices, recv_ghost_offsets,
        neighbor_reverse
end

"""
    _remap_colval(local_colval, owned_range, ghost_global_indices)

Remap global column indices to local numbering.
Owned columns map to `1:n_owned`, ghost columns to `(n_owned+1):(n_owned+n_ghost)`.
"""
function _remap_colval(
        local_colval::AbstractVector{Ti},
        owned_range::UnitRange{Int},
        ghost_global_indices::Vector{Int},
    ) where {Ti <: Integer}
    owned_first = first(owned_range)
    n_owned = length(owned_range)

    ghost_to_local = Dict{Int, Ti}()
    for (i, g) in enumerate(ghost_global_indices)
        ghost_to_local[g] = Ti(n_owned + i)
    end

    remapped = similar(local_colval)
    for i in eachindex(local_colval)
        col = Int(local_colval[i])
        if col in owned_range
            remapped[i] = Ti(col - owned_first + 1)
        else
            remapped[i] = ghost_to_local[col]
        end
    end
    return remapped
end

"""
    GhostExchange{Tv,V,VI}

Pre-computed ghost/halo communication topology and GPU buffers for peer-to-peer exchange
between devices. Used by [`scatter!`](@ref) (owner→ghost) and [`reduce!`](@ref) (ghost→owner).

# Fields
- `ghost_global_indices` — global indices needed as ghosts on each device
- `neighbors` — neighboring device indices for each device
- `send_local_indices` — local indices to pack into send buffers per neighbor per device
- `recv_ghost_offsets` — ranges into the ghost region for received data per neighbor per device
- `neighbor_reverse` — precomputed reverse neighbor lookup; `neighbor_reverse[d][k]` is the index of `d` in `neighbors[neighbors[d][k]]`
- `send_buffers` — pre-allocated GPU send buffers per neighbor per device
- `recv_buffers` — pre-allocated GPU receive buffers per neighbor per device
- `send_indices_gpu` — GPU-side copies of send index arrays for gather operations
- `local_x` — per-device extended vectors (`[owned | ghost]`) used during communication
"""
struct GhostExchange{Tv, V <: AbstractVector{Tv}, VI <: AbstractVector{Int}}
    ghost_global_indices::Vector{Vector{Int}}
    neighbors::Vector{Vector{Int}}
    send_local_indices::Vector{Vector{Vector{Int}}}
    recv_ghost_offsets::Vector{Vector{UnitRange{Int}}}
    neighbor_reverse::Vector{Vector{Int}}
    send_buffers::Vector{Vector{V}}
    recv_buffers::Vector{Vector{V}}
    send_indices_gpu::Vector{Vector{VI}}
    local_x::Vector{V}
end

function GhostExchange(
        ghost_global_indices::Vector{Vector{Int}},
        neighbors::Vector{Vector{Int}},
        send_local_indices::Vector{Vector{Vector{Int}}},
        recv_ghost_offsets::Vector{Vector{UnitRange{Int}}},
        neighbor_reverse::Vector{Vector{Int}},
        row_spec::PartitionSpec,
        ::Type{Tv},
    ) where {Tv}
    ndevices = row_spec.ndevices

    send_buffers = Vector{Vector{CuVector{Tv}}}(undef, ndevices)
    recv_buffers = Vector{Vector{CuVector{Tv}}}(undef, ndevices)
    send_indices_gpu = Vector{Vector{CuVector{Int}}}(undef, ndevices)
    local_x = Vector{CuVector{Tv}}(undef, ndevices)

    @sync for d in 1:ndevices
        @async begin
            CUDA.device!(device_id(row_spec, d))
            n_owned = length(row_spec.ranges[d])
            n_ghost = length(ghost_global_indices[d])
            local_x[d] = CuVector{Tv}(undef, n_owned + n_ghost)

            send_buffers[d] = CuVector{Tv}[
                CuVector{Tv}(undef, length(send_local_indices[d][k]))
                    for k in eachindex(neighbors[d])
            ]
            recv_buffers[d] = CuVector{Tv}[
                CuVector{Tv}(undef, length(recv_ghost_offsets[d][k]))
                    for k in eachindex(neighbors[d])
            ]
            send_indices_gpu[d] = CuVector{Int}[
                CuVector{Int}(send_local_indices[d][k])
                    for k in eachindex(neighbors[d])
            ]
        end
    end

    return GhostExchange{Tv, CuVector{Tv}, CuVector{Int}}(
        ghost_global_indices, neighbors, send_local_indices, recv_ghost_offsets,
        neighbor_reverse, send_buffers, recv_buffers, send_indices_gpu, local_x,
    )
end

"""
    GhostExchange(ghost_global_indices, spec, Tv)

Construct a `GhostExchange` from user-specified per-device ghost index lists, independent
of any matrix. Each `ghost_global_indices[d]` is a sorted vector of global indices that
device `d` needs as ghosts.
"""
function GhostExchange(
        ghost_global_indices::AbstractVector{<:AbstractVector{Int}},
        spec::PartitionSpec,
        ::Type{Tv},
    ) where {Tv}
    length(ghost_global_indices) == spec.ndevices || throw(
        ArgumentError(
            "Length of ghost_global_indices ($(length(ghost_global_indices))) must equal ndevices ($(spec.ndevices))"
        ),
    )
    for d in 1:spec.ndevices
        for g in ghost_global_indices[d]
            (1 <= g <= spec.len) || throw(
                BoundsError("Ghost index $g on device $d is out of range 1:$(spec.len)")
            )
            g in spec.ranges[d] && throw(
                ArgumentError(
                    "Ghost index $g on device $d falls in its owned range $(spec.ranges[d])"
                ),
            )
        end
    end

    ggi = [collect(Int, gi) for gi in ghost_global_indices]
    neighbors, send_local_indices, recv_ghost_offsets, neighbor_reverse =
        _compute_ghost_topology(ggi, spec)

    return GhostExchange(
        ggi, neighbors, send_local_indices, recv_ghost_offsets, neighbor_reverse, spec, Tv
    )
end

"""
    copy_exchange(ghost::GhostExchange, spec::PartitionSpec)

Create a new [`GhostExchange`](@ref) that shares the same topology (neighbor lists, index
mappings) but allocates independent GPU communication buffers. Used by `similar` to give
each vector its own scratch space for [`scatter!`](@ref) / [`reduce!`](@ref).
"""
function copy_exchange(ghost::GhostExchange{Tv, V, VI}, spec::PartitionSpec) where {Tv, V, VI}
    ndevices = length(ghost.local_x)
    send_buffers = Vector{Vector{V}}(undef, ndevices)
    recv_buffers = Vector{Vector{V}}(undef, ndevices)
    local_x = Vector{V}(undef, ndevices)

    @sync for d in 1:ndevices
        @async begin
            CUDA.device!(device_id(spec, d))
            local_x[d] = similar(ghost.local_x[d])
            send_buffers[d] = V[similar(b) for b in ghost.send_buffers[d]]
            recv_buffers[d] = V[similar(b) for b in ghost.recv_buffers[d]]
        end
    end

    return GhostExchange{Tv, V, VI}(
        ghost.ghost_global_indices,
        ghost.neighbors,
        ghost.send_local_indices,
        ghost.recv_ghost_offsets,
        ghost.neighbor_reverse,
        send_buffers,
        recv_buffers,
        ghost.send_indices_gpu,
        local_x,
    )
end

copy_exchange(::Nothing, ::PartitionSpec) = nothing

"""
    attach_ghost(v::MultiDeviceVector, ghost::GhostExchange)

Return a new `MultiDeviceVector` that shares the same partition data but carries the given
[`GhostExchange`](@ref), enabling self-contained [`scatter!`](@ref) and [`reduce!`](@ref).
"""
function attach_ghost(
        v::MultiDeviceVector{T, VP, P}, ghost::GhostExchange{T}
    ) where {T, VP, P}
    return MultiDeviceVector{T, VP, P, typeof(ghost)}(v.partitions, v.spec, ghost)
end

"""
    attach_ghost(v::MultiDeviceVector, ghost_global_indices)

Build a [`GhostExchange`](@ref) from per-device ghost index lists and attach it to `v`.
Creates fresh GPU buffers (no aliasing with other exchanges).
"""
function attach_ghost(
        v::MultiDeviceVector{T}, ghost_global_indices::AbstractVector{<:AbstractVector{Int}}
    ) where {T}
    ghost = GhostExchange(ghost_global_indices, v.spec, T)
    return attach_ghost(v, ghost)
end

"""
    scatter!(x::MultiDeviceVector)

Owner→ghost exchange using the vector's own [`GhostExchange`](@ref). The vector must have
been constructed with a ghost exchange (see [`attach_ghost`](@ref)).
"""
function scatter!(x::MultiDeviceVector{Tv, VP, P, GE}) where {Tv, VP, P, GE <: GhostExchange}
    return scatter!(x, x.ghost_exchange, x.spec)
end

function scatter!(::MultiDeviceVector{Tv, VP, P, Nothing}) where {Tv, VP, P}
    throw(
        ArgumentError(
            "scatter! requires a GhostExchange; use attach_ghost(x, ghost) first, or call scatter!(x, ghost, spec)"
        )
    )
end

"""
    scatter!(x, ghost, spec)

Owner→ghost exchange: pack owned data into send buffers, transfer between devices,
and assemble `local_x = [owned | ghost]` on each device.
"""
function scatter!(
        x::MultiDeviceVector{Tv},
        ghost::GhostExchange{Tv},
        row_spec::PartitionSpec,
    ) where {Tv}
    ndevices = row_spec.ndevices

    # Phase 1: Pack send buffers (gather owned values at send indices)
    @sync for d in 1:ndevices
        @async begin
            CUDA.device!(device_id(row_spec, d))
            for k in eachindex(ghost.neighbors[d])
                if !isempty(ghost.send_local_indices[d][k])
                    ghost.send_buffers[d][k] .= x.partitions[d][ghost.send_indices_gpu[d][k]]
                end
            end
        end
    end

    # Phase 2: P2P transfer into recv buffers and assemble local_x
    return @sync for d in 1:ndevices
        @async begin
            CUDA.device!(device_id(row_spec, d))
            n_owned = length(row_spec.ranges[d])

            # Copy owned partition into local_x[1:n_owned]
            copyto!(view(ghost.local_x[d], 1:n_owned), x.partitions[d])

            # Receive ghosts from each neighbor
            for (k, nbr) in enumerate(ghost.neighbors[d])
                offsets = ghost.recv_ghost_offsets[d][k]
                if !isempty(offsets)
                    k_in_nbr = ghost.neighbor_reverse[d][k]
                    copyto!(ghost.recv_buffers[d][k], ghost.send_buffers[nbr][k_in_nbr])
                    copyto!(
                        view(ghost.local_x[d], n_owned .+ offsets), ghost.recv_buffers[d][k]
                    )
                end
            end
        end
    end
end

"""
    reduce!(x::MultiDeviceVector, op)

Ghost→owner reduction using the vector's own [`GhostExchange`](@ref). The vector must have
been constructed with a ghost exchange (see [`attach_ghost`](@ref)).
"""
function reduce!(
        x::MultiDeviceVector{Tv, VP, P, GE}, op::F
    ) where {Tv, VP, P, GE <: GhostExchange, F <: Function}
    return reduce!(x, x.ghost_exchange, x.spec, op)
end

function reduce!(::MultiDeviceVector{Tv, VP, P, Nothing}, ::F) where {Tv, VP, P, F <: Function}
    throw(
        ArgumentError(
            "reduce! requires a GhostExchange; use attach_ghost(x, ghost) first, or call reduce!(x, ghost, spec, op)"
        )
    )
end

"""
    reduce!(x, ghost, spec, op)

Ghost→owner reduction: for each device, copies owned values from `ghost.local_x` into
`x.partitions`, packs ghost contributions into buffers, transfers them to owner devices,
and applies `op` element-wise into the owner's partition of `x`.

The caller is responsible for populating `ghost.local_x[d]` (the `[owned | ghost]` extended
vector) before calling `reduce!`.
"""
function reduce!(
        x::MultiDeviceVector{Tv},
        ghost::GhostExchange{Tv},
        spec::PartitionSpec,
        op::F,
    ) where {Tv, F <: Function}
    ndevices = spec.ndevices

    # Phase 1: Copy owned portion from local_x into x, and pack ghost contributions
    @sync for d in 1:ndevices
        @async begin
            CUDA.device!(device_id(spec, d))
            n_owned = length(spec.ranges[d])

            copyto!(x.partitions[d], view(ghost.local_x[d], 1:n_owned))

            # Pack ghost values into recv_buffers (reused as send direction for reduce)
            for (k, _nbr) in enumerate(ghost.neighbors[d])
                offsets = ghost.recv_ghost_offsets[d][k]
                if !isempty(offsets)
                    copyto!(
                        ghost.recv_buffers[d][k],
                        view(ghost.local_x[d], n_owned .+ offsets),
                    )
                end
            end
        end
    end

    # Phase 2: P2P transfer ghost contributions to owners and apply op
    @sync for d in 1:ndevices
        @async begin
            CUDA.device!(device_id(spec, d))
            for (k, nbr) in enumerate(ghost.neighbors[d])
                if !isempty(ghost.send_local_indices[d][k])
                    k_in_nbr = ghost.neighbor_reverse[d][k]
                    copyto!(ghost.send_buffers[d][k], ghost.recv_buffers[nbr][k_in_nbr])
                    x.partitions[d][ghost.send_indices_gpu[d][k]] .= op.(
                        x.partitions[d][ghost.send_indices_gpu[d][k]],
                        ghost.send_buffers[d][k],
                    )
                end
            end
        end
    end
    return x
end
