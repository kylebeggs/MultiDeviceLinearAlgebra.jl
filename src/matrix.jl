"""
    MultiDeviceSparseMatrixCSR{Tv,Ti,GE,VP,P} <: AbstractMatrix{Tv}

Row-partitioned sparse CSR matrix distributed across CUDA devices. Each device holds its
row block as a `CuSparseMatrixCSR` with locally-remapped column indices. Off-partition column
values are exchanged via the embedded [`GhostExchange`](@ref) before each SpMV.

# Fields
- `partitions::VP` — per-device `CuSparseMatrixCSR` row blocks
- `ghost_exchange::GE` — [`GhostExchange`](@ref) for P2P halo communication
- `row_spec::P` — [`PartitionSpec`](@ref) describing the row distribution
- `dims::Tuple{Int,Int}` — global matrix dimensions `(nrows, ncols)`
"""
struct MultiDeviceSparseMatrixCSR{Tv, Ti, GE, VP <: AbstractVector{<:CuSparseMatrixCSR{Tv, Ti}}, P <: PartitionSpec} <: AbstractMatrix{Tv}
    partitions::VP
    ghost_exchange::GE
    row_spec::P
    dims::Tuple{Int, Int}
end

function MultiDeviceSparseMatrixCSR(
        A::SparseMatrixCSC{Tv, Ti}; ndevices::Int = length(CUDA.devices())
    ) where {Tv, Ti}
    nrows = size(A, 1)
    @assert ndevices <= nrows "More devices ($ndevices) than rows ($nrows)"
    row_spec = compute_partition_ranges(nrows, ndevices)
    return MultiDeviceSparseMatrixCSR(A, row_spec)
end

function MultiDeviceSparseMatrixCSR(
        A::SparseMatrixCSC{Tv, Ti}, row_spec::PartitionSpec
    ) where {Tv, Ti}
    nrows, ncols = size(A)

    validated = PartitionSpec(row_spec.ranges; devices = collect(Int, row_spec.devices))
    validated.len == nrows || throw(
        DimensionMismatch("PartitionSpec covers $(validated.len) rows but matrix has $nrows")
    )
    validated.ndevices == row_spec.ndevices || throw(
        ArgumentError(
            "Inconsistent PartitionSpec: ndevices=$(row_spec.ndevices) but " *
                "length(ranges)=$(validated.ndevices)",
        ),
    )
    row_spec = validated
    ndevices = row_spec.ndevices

    At = SparseMatrixCSC(sparse(A'))
    csr_rowptr = At.colptr
    csr_colval = At.rowval
    csr_nzval = At.nzval

    ghost_global_indices, neighbors, send_local_indices, recv_ghost_offsets,
        neighbor_reverse = _compute_ghost_map(csr_rowptr, csr_colval, row_spec)

    partitions = Vector{CuSparseMatrixCSR{Tv, Ti}}(undef, ndevices)

    @sync for d in 1:ndevices
        @async begin
            CUDA.device!(device_id(row_spec, d))
            r = row_spec.ranges[d]
            local_nrows = length(r)
            n_owned = length(r)
            n_ghost = length(ghost_global_indices[d])

            rp_start = csr_rowptr[first(r)]
            rp_end = csr_rowptr[last(r) + 1] - 1

            local_rowptr = csr_rowptr[first(r):(last(r) + 1)] .- (rp_start - 1)
            local_colval = csr_colval[rp_start:rp_end]
            local_nzval = csr_nzval[rp_start:rp_end]

            remapped_colval = _remap_colval(local_colval, r, ghost_global_indices[d])

            d_rowptr = CuVector{Ti}(local_rowptr)
            d_colval = CuVector{Ti}(remapped_colval)
            d_nzval = CuVector{Tv}(local_nzval)

            local_ncols = n_owned + n_ghost
            partitions[d] = CuSparseMatrixCSR{Tv, Ti}(
                d_rowptr, d_colval, d_nzval, (local_nrows, local_ncols)
            )
        end
    end

    ghost_exchange = GhostExchange(
        ghost_global_indices, neighbors, send_local_indices, recv_ghost_offsets,
        neighbor_reverse, row_spec, Tv,
    )

    return MultiDeviceSparseMatrixCSR{Tv, Ti, typeof(ghost_exchange), typeof(partitions), typeof(row_spec)}(
        partitions, ghost_exchange, row_spec, (nrows, ncols)
    )
end

Base.size(A::MultiDeviceSparseMatrixCSR) = A.dims
Base.size(A::MultiDeviceSparseMatrixCSR, d::Int) = A.dims[d]
Base.eltype(::Type{<:MultiDeviceSparseMatrixCSR{Tv}}) where {Tv} = Tv
