function _empty_mdv(v::MultiDeviceVector{T}) where {T}
    empty_spec = PartitionSpec(
        [1:0 for _ in 1:v.spec.ndevices],
        0,
        v.spec.ndevices,
    )
    partitions = Vector{CuVector{T}}(undef, v.spec.ndevices)
    @sync for d in 1:v.spec.ndevices
        @async begin
            CUDA.device!(d - 1)
            partitions[d] = CuVector{T}(undef, 0)
        end
    end
    return MultiDeviceVector{T}(partitions, empty_spec)
end

function Krylov.CgWorkspace(A::MultiDeviceSparseMatrixCSR{Tv}, b::MultiDeviceVector{Tv}) where {Tv}
    b_empty = _empty_mdv(b)
    kc = Krylov.KrylovConstructor(b; vm_empty=b_empty)
    return Krylov.CgWorkspace(kc)
end

function mdla_solve(A::MultiDeviceSparseMatrixCSR, b::MultiDeviceVector; kwargs...)
    x, stats = Krylov.cg(A, b; kwargs...)
    return x, stats
end
