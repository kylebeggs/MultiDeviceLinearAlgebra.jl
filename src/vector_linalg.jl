function Base.fill!(v::MultiDeviceVector{T}, val) where {T}
    @sync for d in 1:v.spec.ndevices
        @async begin
            CUDA.device!(device_id(v.spec, d))
            fill!(v.partitions[d], val)
        end
    end
    return v
end

function Base.copyto!(dst::MultiDeviceVector{T}, src::MultiDeviceVector{T}) where {T}
    @assert dst.spec.len == src.spec.len "Length mismatch: $(dst.spec.len) vs $(src.spec.len)"
    @assert dst.spec.ndevices == src.spec.ndevices "Device count mismatch"
    @sync for d in 1:dst.spec.ndevices
        @async begin
            CUDA.device!(device_id(dst.spec, d))
            copyto!(dst.partitions[d], src.partitions[d])
        end
    end
    return dst
end

function LinearAlgebra.dot(x::MultiDeviceVector{T}, y::MultiDeviceVector{T}) where {T}
    @assert x.spec.len == y.spec.len "Length mismatch"
    partial = Vector{T}(undef, x.spec.ndevices)
    @sync for d in 1:x.spec.ndevices
        @async begin
            CUDA.device!(device_id(x.spec, d))
            partial[d] = CUDA.CUBLAS.dot(length(x.partitions[d]), x.partitions[d], y.partitions[d])
        end
    end
    return sum(partial)
end

function LinearAlgebra.norm(v::MultiDeviceVector{T}) where {T <: Real}
    partial = Vector{T}(undef, v.spec.ndevices)
    @sync for d in 1:v.spec.ndevices
        @async begin
            CUDA.device!(device_id(v.spec, d))
            partial[d] = CUDA.CUBLAS.dot(length(v.partitions[d]), v.partitions[d], v.partitions[d])
        end
    end
    return sqrt(sum(partial))
end

function LinearAlgebra.norm(v::MultiDeviceVector{T}) where {T <: Complex}
    R = real(T)
    partial = Vector{R}(undef, v.spec.ndevices)
    @sync for d in 1:v.spec.ndevices
        @async begin
            CUDA.device!(device_id(v.spec, d))
            nrm = CUDA.CUBLAS.nrm2(length(v.partitions[d]), v.partitions[d])
            partial[d] = nrm * nrm
        end
    end
    return sqrt(sum(partial))
end

function LinearAlgebra.axpy!(α::Number, x::MultiDeviceVector{T}, y::MultiDeviceVector{T}) where {T}
    @assert x.spec.len == y.spec.len "Length mismatch"
    @sync for d in 1:x.spec.ndevices
        @async begin
            CUDA.device!(device_id(x.spec, d))
            CUDA.CUBLAS.axpy!(length(x.partitions[d]), T(α), x.partitions[d], y.partitions[d])
        end
    end
    return y
end

function LinearAlgebra.axpby!(α::Number, x::MultiDeviceVector{T}, β::Number, y::MultiDeviceVector{T}) where {T}
    @assert x.spec.len == y.spec.len "Length mismatch"
    @sync for d in 1:x.spec.ndevices
        @async begin
            CUDA.device!(device_id(x.spec, d))
            CUDA.CUBLAS.scal!(length(y.partitions[d]), T(β), y.partitions[d])
            CUDA.CUBLAS.axpy!(length(x.partitions[d]), T(α), x.partitions[d], y.partitions[d])
        end
    end
    return y
end

function LinearAlgebra.rmul!(v::MultiDeviceVector{T}, s::Number) where {T}
    @sync for d in 1:v.spec.ndevices
        @async begin
            CUDA.device!(device_id(v.spec, d))
            CUDA.CUBLAS.scal!(length(v.partitions[d]), T(s), v.partitions[d])
        end
    end
    return v
end

function LinearAlgebra.lmul!(s::Number, v::MultiDeviceVector{T}) where {T}
    return rmul!(v, s)
end
