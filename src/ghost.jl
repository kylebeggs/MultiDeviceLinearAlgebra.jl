"""
    _compute_ghost_map(csr_rowptr, csr_colval, row_spec)

Compute ghost communication topology for a row-partitioned CSR matrix.
Returns per-device ghost indices, neighbor lists, send indices, and receive offsets.
"""
function _compute_ghost_map(
    csr_rowptr::AbstractVector{Ti},
    csr_colval::AbstractVector{Ti},
    row_spec::PartitionSpec,
) where {Ti<:Integer}
    ndevices = row_spec.ndevices

    ghost_global_indices = Vector{Vector{Int}}(undef, ndevices)
    # ghost_by_owner[d][owner] = sorted global indices that device d needs from owner
    ghost_by_owner = [Dict{Int,Vector{Int}}() for _ in 1:ndevices]

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

        ghosts = sort!(collect(ghost_set))
        ghost_global_indices[d] = ghosts

        for g in ghosts
            owner, _ = device_for_index(row_spec, g)
            owner_list = get!(Vector{Int}, ghost_by_owner[d], owner)
            push!(owner_list, g)
        end
    end

    # needs_from[d][requester] = global indices that requester needs from device d
    needs_from = [Dict{Int,Vector{Int}}() for _ in 1:ndevices]
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
        owned_first = first(row_spec.ranges[d])
        send_local_indices[d] = Vector{Vector{Int}}(undef, length(neighbors[d]))
        for (k, nbr) in enumerate(neighbors[d])
            globals_needed = get(needs_from[d], nbr, Int[])
            send_local_indices[d][k] = [g - owned_first + 1 for g in globals_needed]
        end
    end

    return ghost_global_indices, neighbors, send_local_indices, recv_ghost_offsets
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
) where {Ti<:Integer}
    owned_first = first(owned_range)
    n_owned = length(owned_range)

    ghost_to_local = Dict{Int,Ti}()
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
between devices during sparse matrix-vector multiplication. Before each SpMV, [`consistent!`](@ref)
uses this structure to exchange off-partition column values between neighboring devices.

# Fields
- `ghost_global_indices` — global column indices needed as ghosts on each device
- `neighbors` — neighboring device indices for each device
- `send_local_indices` — local indices to pack into send buffers per neighbor per device
- `recv_ghost_offsets` — ranges into the ghost region for received data per neighbor per device
- `send_buffers` — pre-allocated GPU send buffers per neighbor per device
- `recv_buffers` — pre-allocated GPU receive buffers per neighbor per device
- `send_indices_gpu` — GPU-side copies of send index arrays for gather operations
- `local_x` — per-device extended vectors (`[owned | ghost]`) used during SpMV
"""
struct GhostExchange{Tv,V<:AbstractVector{Tv},VI<:AbstractVector{Int}}
    ghost_global_indices::Vector{Vector{Int}}
    neighbors::Vector{Vector{Int}}
    send_local_indices::Vector{Vector{Vector{Int}}}
    recv_ghost_offsets::Vector{Vector{UnitRange{Int}}}
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

    return GhostExchange{Tv,CuVector{Tv},CuVector{Int}}(
        ghost_global_indices, neighbors, send_local_indices, recv_ghost_offsets,
        send_buffers, recv_buffers, send_indices_gpu, local_x,
    )
end

"""
    consistent!(x, ghost, row_spec)

Exchange ghost values: pack owned data into send buffers, transfer between devices,
and assemble `local_x = [owned | ghost]` on each device.
"""
function consistent!(
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
    @sync for d in 1:ndevices
        @async begin
            CUDA.device!(device_id(row_spec, d))
            n_owned = length(row_spec.ranges[d])

            # Copy owned partition into local_x[1:n_owned]
            copyto!(view(ghost.local_x[d], 1:n_owned), x.partitions[d])

            # Receive ghosts from each neighbor
            for (k, nbr) in enumerate(ghost.neighbors[d])
                offsets = ghost.recv_ghost_offsets[d][k]
                if !isempty(offsets)
                    k_in_nbr = findfirst(==(d), ghost.neighbors[nbr])
                    copyto!(ghost.recv_buffers[d][k], ghost.send_buffers[nbr][k_in_nbr])
                    copyto!(
                        view(ghost.local_x[d], n_owned .+ offsets), ghost.recv_buffers[d][k]
                    )
                end
            end
        end
    end
end
