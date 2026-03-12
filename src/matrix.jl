struct MultiDeviceSparseMatrixCSR{Tv,Ti,GE} <: AbstractMatrix{Tv}
    partitions::Vector{CuSparseMatrixCSR{Tv,Ti}}
    ghost_exchange::GE
    row_spec::PartitionSpec
    dims::Tuple{Int,Int}
end

function MultiDeviceSparseMatrixCSR(
    A::SparseMatrixCSC{Tv,Ti}; ndevices::Int=length(CUDA.devices())
) where {Tv,Ti}
    nrows, ncols = size(A)
    @assert ndevices <= nrows "More devices ($ndevices) than rows ($nrows)"

    At = SparseMatrixCSC(sparse(A'))
    csr_rowptr = At.colptr
    csr_colval = At.rowval
    csr_nzval = At.nzval

    row_spec = compute_partition_ranges(nrows, ndevices)

    ghost_global_indices, neighbors, send_local_indices, recv_ghost_offsets =
        _compute_ghost_map(csr_rowptr, csr_colval, row_spec)

    partitions = Vector{CuSparseMatrixCSR{Tv,Ti}}(undef, ndevices)

    @sync for d in 1:ndevices
        @async begin
            CUDA.device!(d - 1)
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
            partitions[d] = CuSparseMatrixCSR{Tv,Ti}(
                d_rowptr, d_colval, d_nzval, (local_nrows, local_ncols)
            )
        end
    end

    ghost_exchange = GhostExchange(
        ghost_global_indices, neighbors, send_local_indices, recv_ghost_offsets,
        row_spec, Tv,
    )

    return MultiDeviceSparseMatrixCSR{Tv,Ti,typeof(ghost_exchange)}(
        partitions, ghost_exchange, row_spec, (nrows, ncols)
    )
end

Base.size(A::MultiDeviceSparseMatrixCSR) = A.dims
Base.size(A::MultiDeviceSparseMatrixCSR, d::Int) = A.dims[d]
Base.eltype(::Type{<:MultiDeviceSparseMatrixCSR{Tv}}) where {Tv} = Tv
