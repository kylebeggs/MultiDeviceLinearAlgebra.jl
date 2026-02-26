function _allgather_x!(buffers::Vector{CuVector{Tv}}, x::MultiDeviceVector{Tv}) where {Tv}
    ndevices = x.spec.ndevices
    @sync for d in 1:ndevices
        @async begin
            CUDA.device!(d - 1)
            for s in 1:ndevices
                r = x.spec.ranges[s]
                copyto!(view(buffers[d], r), x.partitions[s])
            end
        end
    end
end

function LinearAlgebra.mul!(
    y::MultiDeviceVector{Tv},
    A::MultiDeviceSparseMatrixCSR{Tv},
    x::MultiDeviceVector{Tv},
) where {Tv}
    @assert y.spec.len == A.dims[1] "y length $(y.spec.len) != A rows $(A.dims[1])"
    @assert x.spec.len == A.dims[2] "x length $(x.spec.len) != A cols $(A.dims[2])"

    _allgather_x!(A.x_buffers, x)

    @sync for d in 1:x.spec.ndevices
        @async begin
            CUDA.device!(d - 1)
            mul!(y.partitions[d], A.partitions[d], A.x_buffers[d])
        end
    end
    return y
end

function LinearAlgebra.mul!(
    y::MultiDeviceVector{Tv},
    A::MultiDeviceSparseMatrixCSR{Tv},
    x::MultiDeviceVector{Tv},
    α::Number,
    β::Number,
) where {Tv}
    @assert y.spec.len == A.dims[1] "y length $(y.spec.len) != A rows $(A.dims[1])"
    @assert x.spec.len == A.dims[2] "x length $(x.spec.len) != A cols $(A.dims[2])"

    _allgather_x!(A.x_buffers, x)

    @sync for d in 1:x.spec.ndevices
        @async begin
            CUDA.device!(d - 1)
            mul!(y.partitions[d], A.partitions[d], A.x_buffers[d], Tv(α), Tv(β))
        end
    end
    return y
end
