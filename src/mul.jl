"""
    _check_spmv_partitions(y, A, x)

Reject an SpMV whose vectors are not partitioned like the matrix.

Matching lengths are not enough: `scatter!` assembles each device's `[owned | ghost]` slice
from `x.partitions[d]` assuming `x` is split exactly like `A`'s columns, so an `x` of the right
total length but a different partition reads the wrong entries and produces a plausible wrong
answer instead of an error.
"""
function _check_spmv_partitions(
        y::MultiDeviceVector, A::MultiDeviceSparseMatrixCSR, x::MultiDeviceVector
    )
    _specs_match(x.spec, A.col_spec) || throw(
        ArgumentError(
            "x must be partitioned like A's columns: A.col_spec has ranges " *
                "$(A.col_spec.ranges) on devices $(collect(Int, A.col_spec.devices)), x has " *
                "ranges $(x.spec.ranges) on devices $(collect(Int, x.spec.devices))",
        ),
    )
    _specs_match(y.spec, A.row_spec) || throw(
        ArgumentError(
            "y must be partitioned like A's rows: A.row_spec has ranges " *
                "$(A.row_spec.ranges) on devices $(collect(Int, A.row_spec.devices)), y has " *
                "ranges $(y.spec.ranges) on devices $(collect(Int, y.spec.devices))",
        ),
    )
    return nothing
end

function LinearAlgebra.mul!(
        y::MultiDeviceVector{Tv},
        A::MultiDeviceSparseMatrixCSR{Tv},
        x::MultiDeviceVector{Tv},
    ) where {Tv}
    _check_spmv_partitions(y, A, x)

    scatter!(x, A.ghost_exchange, A.col_spec)

    # Fresh tasks, so the SpMV reads `local_x[d]` from a different stream than the one
    # `scatter!` assembled it on. Ordered by the same CUDA.jl mechanism documented at the
    # phase boundary in `src/ghost.jl`.
    @sync for d in 1:A.row_spec.ndevices
        @async begin
            CUDA.device!(device_id(A.row_spec, d))
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
    _check_spmv_partitions(y, A, x)

    scatter!(x, A.ghost_exchange, A.col_spec)

    @sync for d in 1:A.row_spec.ndevices
        @async begin
            CUDA.device!(device_id(A.row_spec, d))
            mul!(y.partitions[d], A.partitions[d], A.ghost_exchange.local_x[d], Tv(α), Tv(β))
        end
    end
    return y
end
