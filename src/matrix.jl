"""
    DeviceIndex

Index type of the CSR arrays held on device.

Fixed at `Int32` rather than following the host matrix's index type, because cuSPARSE's
sparse-sparse routines — SpGEMM, `csr2csc`, `geam` — accept only 32-bit indices. SpMV is the
exception: its generic descriptor honours whatever `eltype(rowPtr)` it is handed, which is why
64-bit indices worked here for as long as SpMV was the only operation. A single device block
overflows this only past `typemax(Int32)` nonzeros — about 26 GB of `Float64` CSR, within
reach of an 80 GB device — so construction checks and rejects it.
"""
const DeviceIndex = Int32

"""
    MultiDeviceSparseMatrixCSR{Tv,Ti,GE,VP,P,PC} <: AbstractMatrix{Tv}

Row-partitioned sparse CSR matrix distributed across CUDA devices. Each device holds its
row block as a `CuSparseMatrixCSR` with locally-remapped column indices. Off-partition column
values are exchanged via the embedded [`GhostExchange`](@ref) before each SpMV.

Rows and columns are partitioned independently, so the matrix may be rectangular. For a square
matrix built from a single `PartitionSpec` the two are the same object, and `col_spec ===
row_spec` holds.

# Fields
- `partitions::VP` — per-device `CuSparseMatrixCSR` row blocks
- `ghost_exchange::GE` — [`GhostExchange`](@ref) for P2P halo communication, built against `col_spec`
- `row_spec::P` — [`PartitionSpec`](@ref) describing the row distribution
- `col_spec::PC` — [`PartitionSpec`](@ref) describing the column distribution
- `dims::Tuple{Int,Int}` — global matrix dimensions `(nrows, ncols)`
"""
struct MultiDeviceSparseMatrixCSR{Tv, Ti, GE, VP <: AbstractVector{<:CuSparseMatrixCSR{Tv, Ti}}, P <: PartitionSpec, PC <: PartitionSpec} <: AbstractMatrix{Tv}
    partitions::VP
    ghost_exchange::GE
    row_spec::P
    col_spec::PC
    dims::Tuple{Int, Int}
end

"""
    _revalidate_spec(spec, what)

Re-run the validating `PartitionSpec` constructor over `spec`'s ranges, so a spec built through
the unchecked inner constructor is rejected here rather than corrupting a partition later.
`what` names the spec in the error message.
"""
function _revalidate_spec(spec::PartitionSpec, what::AbstractString)
    validated = PartitionSpec(spec.ranges; devices = collect(Int, spec.devices))
    validated.ndevices == spec.ndevices || throw(
        ArgumentError(
            "Inconsistent $what PartitionSpec: ndevices=$(spec.ndevices) but " *
                "length(ranges)=$(validated.ndevices)",
        ),
    )
    return validated
end

function MultiDeviceSparseMatrixCSR(
        # `Int(...)` is load-bearing: `length(CUDA.devices())` is an `Int32` and a typed
        # keyword default is not converted, so without it the no-kwarg call MethodErrors.
        A::SparseMatrixCSC{Tv, Ti}; ndevices::Int = Int(length(CUDA.devices()))
    ) where {Tv, Ti}
    nrows, ncols = size(A)
    @assert ndevices <= nrows "More devices ($ndevices) than rows ($nrows)"
    @assert ndevices <= ncols "More devices ($ndevices) than columns ($ncols)"
    row_spec = compute_partition_ranges(nrows, ndevices)
    # Sharing the object in the square case is what makes `A.col_spec === A.row_spec` hold, so
    # callers can tell a square matrix from a coincidentally-equal rectangular one.
    col_spec = nrows == ncols ? row_spec : compute_partition_ranges(ncols, ndevices)
    return MultiDeviceSparseMatrixCSR(A, row_spec, col_spec)
end

function MultiDeviceSparseMatrixCSR(
        A::SparseMatrixCSC{Tv, Ti}, row_spec::PartitionSpec
    ) where {Tv, Ti}
    nrows, ncols = size(A)
    nrows == ncols || throw(
        DimensionMismatch(
            "MultiDeviceSparseMatrixCSR(A, row_spec) partitions columns like rows, which only " *
                "makes sense for a square matrix; A is $(nrows)×$(ncols) — pass an explicit " *
                "column partition instead",
        ),
    )
    return MultiDeviceSparseMatrixCSR(A, row_spec, row_spec)
end

function MultiDeviceSparseMatrixCSR(
        A::SparseMatrixCSC{Tv, Ti}, row_spec::PartitionSpec, col_spec::PartitionSpec
    ) where {Tv, Ti}
    nrows, ncols = size(A)

    shared_spec = col_spec === row_spec
    row_spec = _revalidate_spec(row_spec, "row")
    col_spec = shared_spec ? row_spec : _revalidate_spec(col_spec, "column")

    row_spec.len == nrows || throw(
        DimensionMismatch("Row PartitionSpec covers $(row_spec.len) rows but matrix has $nrows")
    )
    col_spec.len == ncols || throw(
        DimensionMismatch(
            "Column PartitionSpec covers $(col_spec.len) columns but matrix has $ncols"
        )
    )
    row_spec.ndevices == col_spec.ndevices || throw(
        ArgumentError(
            "Row partition spans $(row_spec.ndevices) devices but column partition spans " *
                "$(col_spec.ndevices)",
        ),
    )
    collect(Int, row_spec.devices) == collect(Int, col_spec.devices) || throw(
        ArgumentError(
            "Row and column partitions must use the same CUDA devices in the same order, got " *
                "$(collect(Int, row_spec.devices)) and $(collect(Int, col_spec.devices))",
        ),
    )
    ndevices = row_spec.ndevices

    At = SparseMatrixCSC(sparse(A'))
    csr_rowptr = At.colptr
    csr_colval = At.rowval
    csr_nzval = At.nzval

    _check_device_index_range(csr_rowptr, row_spec, ncols)

    ghost_global_indices, neighbors, send_local_indices, recv_ghost_offsets,
        neighbor_reverse = _compute_ghost_map(csr_rowptr, csr_colval, row_spec, col_spec)

    partitions = Vector{CuSparseMatrixCSR{Tv, DeviceIndex}}(undef, ndevices)

    @sync for d in 1:ndevices
        @async begin
            CUDA.device!(device_id(row_spec, d))
            r = row_spec.ranges[d]
            c = col_spec.ranges[d]
            local_nrows = length(r)
            n_owned = length(c)
            n_ghost = length(ghost_global_indices[d])

            rp_start = csr_rowptr[first(r)]
            rp_end = csr_rowptr[last(r) + 1] - 1

            local_rowptr = csr_rowptr[first(r):(last(r) + 1)] .- (rp_start - 1)
            local_colval = csr_colval[rp_start:rp_end]
            local_nzval = csr_nzval[rp_start:rp_end]

            remapped_colval = _remap_colval(local_colval, c, ghost_global_indices[d])

            d_rowptr = CuVector{DeviceIndex}(convert(Vector{DeviceIndex}, local_rowptr))
            d_colval = CuVector{DeviceIndex}(convert(Vector{DeviceIndex}, remapped_colval))
            d_nzval = CuVector{Tv}(local_nzval)

            local_ncols = n_owned + n_ghost
            partitions[d] = CuSparseMatrixCSR{Tv, DeviceIndex}(
                d_rowptr, d_colval, d_nzval, (local_nrows, local_ncols)
            )
        end
    end

    ghost_exchange = GhostExchange(
        ghost_global_indices, neighbors, send_local_indices, recv_ghost_offsets,
        neighbor_reverse, col_spec, Tv,
    )

    return MultiDeviceSparseMatrixCSR{Tv, DeviceIndex, typeof(ghost_exchange), typeof(partitions), typeof(row_spec), typeof(col_spec)}(
        partitions, ghost_exchange, row_spec, col_spec, (nrows, ncols)
    )
end

"""
    _check_device_index_range(csr_rowptr, row_spec, ncols)

Reject a matrix whose device-resident CSR arrays would not fit in [`DeviceIndex`](@ref),
before any of it reaches a device. Column indices are bounded by `ncols`; row pointers by the
nonzero count of the largest row block.
"""
function _check_device_index_range(
        csr_rowptr::AbstractVector{<:Integer}, row_spec::PartitionSpec, ncols::Int
    )
    limit = Int(typemax(DeviceIndex))
    ncols <= limit || throw(
        ArgumentError(
            "Device column indices are stored as $DeviceIndex, so a matrix may have at most " *
                "$limit columns; got $ncols",
        ),
    )
    for d in 1:row_spec.ndevices
        r = row_spec.ranges[d]
        block_nnz = Int(csr_rowptr[last(r) + 1]) - Int(csr_rowptr[first(r)])
        block_nnz < limit || throw(
            ArgumentError(
                "Device $d would hold $block_nnz nonzeros, whose row pointers overflow the " *
                    "$DeviceIndex indices stored on device (limit $(limit - 1))",
            ),
        )
    end
    return nothing
end

"""
    _assemble_from_blocks(blocks, row_spec, col_spec, dims)

Build a [`MultiDeviceSparseMatrixCSR`](@ref) from per-device CSR row blocks whose column
indices are **global**: discover each device's ghost columns, build the matching
[`GhostExchange`](@ref), and remap the columns into local `[owned | ghost]` numbering.

This is the assembly half of every operation that produces a distributed matrix on device
rather than from a host `SparseMatrixCSC`. `blocks[d]` must live on `device_id(col_spec, d)`
and cover `row_spec.ranges[d]`; its `rowPtr` and `nzVal` are adopted, not copied.

Runs as three flat phases and never nests `@sync`, because the `GhostExchange` constructor
between them runs a device loop of its own.
"""
function _assemble_from_blocks(
        blocks::AbstractVector{<:CuSparseMatrixCSR{Tv, DeviceIndex}},
        row_spec::PartitionSpec,
        col_spec::PartitionSpec,
        dims::Tuple{Int, Int},
    ) where {Tv}
    ndevices = row_spec.ndevices
    length(blocks) == ndevices || throw(
        ArgumentError("Got $(length(blocks)) blocks for $ndevices devices")
    )
    ncols = dims[2]

    # Phase 1: discover the ghost columns each device references.
    ghost_global_indices = Vector{Vector{Int}}(undef, ndevices)
    @sync for d in 1:ndevices
        @async begin
            CUDA.device!(device_id(col_spec, d))
            present = CUDA.zeros(Bool, ncols)
            _mark_present!(present, blocks[d].colVal)
            found = Vector{Int}(findall(present))
            CUDA.unsafe_free!(present)
            owned = col_spec.ranges[d]
            # `findall` returns ascending indices, so the ghost list is sorted already — the
            # ordering the ghost topology depends on.
            ghost_global_indices[d] = filter(!in(owned), found)
        end
    end

    # Phase 2: build the exchange (host-side; allocates on every device itself).
    ghost_exchange = GhostExchange(ghost_global_indices, col_spec, Tv)

    # Phase 3: rewrite the global column indices into local numbering.
    partitions = Vector{CuSparseMatrixCSR{Tv, DeviceIndex}}(undef, ndevices)
    @sync for d in 1:ndevices
        @async begin
            CUDA.device!(device_id(col_spec, d))
            owned = col_spec.ranges[d]
            ghosts = CuVector{DeviceIndex}(convert(Vector{DeviceIndex}, ghost_global_indices[d]))
            local_colval = similar(blocks[d].colVal)
            _localize_colval!(local_colval, blocks[d].colVal, owned, ghosts)
            partitions[d] = CuSparseMatrixCSR{Tv, DeviceIndex}(
                blocks[d].rowPtr, local_colval, blocks[d].nzVal,
                (length(row_spec.ranges[d]), length(owned) + length(ghost_global_indices[d])),
            )
        end
    end

    return MultiDeviceSparseMatrixCSR{Tv, DeviceIndex, typeof(ghost_exchange), typeof(partitions), typeof(row_spec), typeof(col_spec)}(
        partitions, ghost_exchange, row_spec, col_spec, dims
    )
end

Base.size(A::MultiDeviceSparseMatrixCSR) = A.dims
Base.size(A::MultiDeviceSparseMatrixCSR, d::Int) = A.dims[d]
Base.eltype(::Type{<:MultiDeviceSparseMatrixCSR{Tv}}) where {Tv} = Tv
