struct MultiDeviceVectorStyle <: Broadcast.AbstractArrayStyle{1} end

MultiDeviceVectorStyle(::Val{1}) = MultiDeviceVectorStyle()
MultiDeviceVectorStyle(::Val{N}) where {N} = Broadcast.DefaultArrayStyle{N}()

Base.BroadcastStyle(::Type{<:MultiDeviceVector}) = MultiDeviceVectorStyle()

function Base.similar(bc::Broadcast.Broadcasted{MultiDeviceVectorStyle}, ::Type{T}) where {T}
    mdv = _find_mdv(bc)
    return MultiDeviceVector{T}(undef, mdv.spec)
end

function _find_mdv(bc::Broadcast.Broadcasted)
    return _find_mdv(bc.args)
end

function _find_mdv(args::Tuple)
    found = _find_mdv(args[1])
    found !== nothing && return found
    return _find_mdv(Base.tail(args))
end

_find_mdv(::Tuple{}) = nothing
_find_mdv(v::MultiDeviceVector) = v
_find_mdv(::Any) = nothing

function _find_mdv(ref::Base.RefValue)
    return _find_mdv(ref[])
end

function Base.copyto!(dest::MultiDeviceVector, bc::Broadcast.Broadcasted{MultiDeviceVectorStyle})
    flat = Broadcast.flatten(bc)
    @sync for d in 1:dest.spec.ndevices
        @async begin
            CUDA.device!(d - 1)
            local_bc = _localize_broadcast(flat, d)
            copyto!(dest.partitions[d], local_bc)
        end
    end
    return dest
end

function _localize_broadcast(bc::Broadcast.Broadcasted, d::Int)
    local_args = map(arg -> _localize_arg(arg, d), bc.args)
    return Broadcast.Broadcasted(bc.f, local_args)
end

_localize_arg(v::MultiDeviceVector, d::Int) = v.partitions[d]
_localize_arg(x, ::Int) = x
_localize_arg(ref::Base.RefValue, ::Int) = ref
