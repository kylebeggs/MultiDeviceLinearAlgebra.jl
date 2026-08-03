function gather(v::MultiDeviceVector{T}) where {T}
    result = Vector{T}(undef, v.spec.len)
    for d in 1:v.spec.ndevices
        CUDA.device!(device_id(v.spec, d))
        result[v.spec.ranges[d]] = Array(v.partitions[d])
    end
    return result
end

"""
    gather(A::MultiDeviceSparseMatrixCSR) → SparseMatrixCSC

Transfer a distributed matrix back to the CPU, undoing the local `[owned | ghost]` column
numbering each device holds. Indices come back as host-native `Int`, not the `Int32` the device
arrays use (see [`DeviceIndex`](@ref)).
"""
function gather(A::MultiDeviceSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    nrows, ncols = A.dims
    I_indices = Int[]
    J_indices = Int[]
    V_values = Tv[]

    for d in 1:A.row_spec.ndevices
        CUDA.device!(device_id(A.row_spec, d))
        part = A.partitions[d]
        h_rowptr = Vector{Ti}(part.rowPtr)
        h_colval = Vector{Ti}(part.colVal)
        h_nzval = Vector{Tv}(part.nzVal)

        row_offset = first(A.row_spec.ranges[d]) - 1
        # Columns are un-remapped against the *column* partition, which is a different range
        # from the row one whenever the matrix is rectangular.
        owned_range = A.col_spec.ranges[d]
        n_owned = length(owned_range)
        ghosts = A.ghost_exchange.ghost_global_indices[d]
        local_nrows = length(A.row_spec.ranges[d])

        for row in 1:local_nrows
            for idx in h_rowptr[row]:(h_rowptr[row + 1] - 1)
                push!(I_indices, row + row_offset)
                local_col = Int(h_colval[idx])
                if local_col <= n_owned
                    global_col = local_col + first(owned_range) - 1
                else
                    global_col = ghosts[local_col - n_owned]
                end
                push!(J_indices, global_col)
                push!(V_values, h_nzval[idx])
            end
        end
    end

    return sparse(I_indices, J_indices, V_values, nrows, ncols)
end
