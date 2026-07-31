function LinearAlgebra.mul!(
        y::MultiDeviceVector{Tv},
        A::MultiDeviceSparseMatrixCSR{Tv},
        x::MultiDeviceVector{Tv},
    ) where {Tv}
    @assert y.spec.len == A.dims[1] "y length $(y.spec.len) != A rows $(A.dims[1])"
    @assert x.spec.len == A.dims[2] "x length $(x.spec.len) != A cols $(A.dims[2])"

    scatter!(x, A.ghost_exchange, A.row_spec)

    # Fresh tasks, so the SpMV reads `local_x[d]` from a different stream than the one
    # `scatter!` assembled it on. Ordered by the same CUDA.jl mechanism documented at the
    # phase boundary in `src/ghost.jl`.
    @sync for d in 1:x.spec.ndevices
        @async begin
            CUDA.device!(device_id(x.spec, d))
            mul!(y.partitions[d], A.partitions[d], A.ghost_exchange.local_x[d])
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

    scatter!(x, A.ghost_exchange, A.row_spec)

    @sync for d in 1:x.spec.ndevices
        @async begin
            CUDA.device!(device_id(x.spec, d))
            mul!(y.partitions[d], A.partitions[d], A.ghost_exchange.local_x[d], Tv(α), Tv(β))
        end
    end
    return y
end
