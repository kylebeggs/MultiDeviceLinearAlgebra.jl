function gather(v::MultiDeviceVector{T}) where {T}
    result = Vector{T}(undef, v.spec.len)
    for d in 1:v.spec.ndevices
        CUDA.device!(d - 1)
        result[v.spec.ranges[d]] = Array(v.partitions[d])
    end
    return result
end

function gather(A::MultiDeviceSparseMatrixCSR{Tv,Ti}) where {Tv,Ti}
    nrows, ncols = A.dims
    I_indices = Ti[]
    J_indices = Ti[]
    V_values = Tv[]

    for d in 1:A.row_spec.ndevices
        CUDA.device!(d - 1)
        part = A.partitions[d]
        h_rowptr = Vector{Ti}(part.rowPtr)
        h_colval = Vector{Ti}(part.colVal)
        h_nzval = Vector{Tv}(part.nzVal)

        row_offset = first(A.row_spec.ranges[d]) - 1
        local_nrows = length(A.row_spec.ranges[d])

        for row in 1:local_nrows
            for idx in h_rowptr[row]:(h_rowptr[row + 1] - 1)
                push!(I_indices, Ti(row + row_offset))
                push!(J_indices, h_colval[idx])
                push!(V_values, h_nzval[idx])
            end
        end
    end

    return sparse(I_indices, J_indices, V_values, nrows, ncols)
end
