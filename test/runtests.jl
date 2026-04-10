using MultiDeviceLinearAlgebra
using Test
using LinearAlgebra
using SparseArrays
using Krylov
using CUDA

const HAS_CUDA = CUDA.functional()
const NGPUS = HAS_CUDA ? length(CUDA.devices()) : 0

# CPU-only tests
include("test_partition.jl")
include("test_ghost.jl")

# Poisson matrix construction tests (CPU portion always runs, GPU portion gated internally)
include("test_poisson.jl")

# GPU tests require CUDA
"""Helper: build neighbor-boundary ghost indices for testing."""
function _neighbor_ghost_indices(spec)
    ggi = Vector{Vector{Int}}(undef, spec.ndevices)
    for d in 1:spec.ndevices
        ghosts = Int[]
        d > 1 && push!(ghosts, last(spec.ranges[d - 1]))
        d < spec.ndevices && push!(ghosts, first(spec.ranges[d + 1]))
        ggi[d] = sort!(ghosts)
    end
    return ggi
end

if HAS_CUDA && NGPUS >= 1
    @info "Running GPU tests with $NGPUS device(s)"
    include("test_ghost_exchange.jl")
    include("test_vector.jl")
    include("test_broadcast.jl")
    include("test_matrix.jl")
    include("test_krylov.jl")
else
    @warn "CUDA not available or no GPUs detected, skipping GPU tests"
end
