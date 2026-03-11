struct MultiDeviceSparseMatrixCSR{Tv,Ti} <: AbstractMatrix{Tv}
    partitions::Vector{CuSparseMatrixCSR{Tv,Ti}}
    x_buffers::Vector{CuVector{Tv}}
    host_x::Vector{Tv}
    row_spec::PartitionSpec
    dims::Tuple{Int,Int}
end

function MultiDeviceSparseMatrixCSR(A::SparseMatrixCSC{Tv,Ti}; ndevices::Int=length(CUDA.devices())) where {Tv,Ti}
    nrows, ncols = size(A)
    @assert ndevices <= nrows "More devices ($ndevices) than rows ($nrows)"

    At = SparseMatrixCSC(sparse(A'))
    csr_rowptr = At.colptr
    csr_colval = At.rowval
    csr_nzval = At.nzval

    row_spec = compute_partition_ranges(nrows, ndevices)

    partitions = Vector{CuSparseMatrixCSR{Tv,Ti}}(undef, ndevices)
    x_buffers = Vector{CuVector{Tv}}(undef, ndevices)

    @sync for d in 1:ndevices
        @async begin
            CUDA.device!(d - 1)
            r = row_spec.ranges[d]
            local_nrows = length(r)

            rp_start = csr_rowptr[first(r)]
            rp_end = csr_rowptr[last(r) + 1] - 1

            local_rowptr = csr_rowptr[first(r):(last(r) + 1)] .- (rp_start - 1)
            local_colval = csr_colval[rp_start:rp_end]
            local_nzval = csr_nzval[rp_start:rp_end]

            local_nnz = length(local_nzval)

            d_rowptr = CuVector{Ti}(local_rowptr)
            d_colval = CuVector{Ti}(local_colval)
            d_nzval = CuVector{Tv}(local_nzval)

            partitions[d] = CuSparseMatrixCSR{Tv,Ti}(
                d_rowptr, d_colval, d_nzval, (local_nrows, ncols)
            )

            x_buffers[d] = CuVector{Tv}(undef, ncols)
        end
    end

    host_x = Vector{Tv}(undef, ncols)

    return MultiDeviceSparseMatrixCSR{Tv,Ti}(partitions, x_buffers, host_x, row_spec, (nrows, ncols))
end

Base.size(A::MultiDeviceSparseMatrixCSR) = A.dims
Base.size(A::MultiDeviceSparseMatrixCSR, d::Int) = A.dims[d]
Base.eltype(::Type{MultiDeviceSparseMatrixCSR{Tv,Ti}}) where {Tv,Ti} = Tv
