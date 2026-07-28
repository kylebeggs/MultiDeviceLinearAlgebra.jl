using MultiDeviceLinearAlgebra
using Test
using LinearAlgebra
using SparseArrays
using Krylov
using CUDA

const HAS_CUDA = CUDA.functional()
const NGPUS = HAS_CUDA ? length(CUDA.devices()) : 0

"""
Device counts swept by the GPU testsets: the small regression points plus the
production default, which is every visible GPU (`src/vector.jl`, `src/matrix.jl`
both default `ndevices = length(CUDA.devices())`). Testing only 1:4 leaves the
configuration users actually get unexercised.
"""
const DEVICE_COUNTS = HAS_CUDA ? unique(filter(<=(NGPUS), [1, 2, 4, NGPUS])) : Int[]

"""
    _numa_node(dev::Int) -> Int

NUMA node of CUDA device `dev` (0-indexed), read from sysfs via the device's PCI
address. Returns `-1` when the node is unknown — non-Linux, sysfs absent, or a
kernel that reports no affinity.
"""
function _numa_node(dev::Int)
    return try
        d = CuDevice(dev)
        domain = CUDA.attribute(d, CUDA.DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)
        bus = CUDA.attribute(d, CUDA.DEVICE_ATTRIBUTE_PCI_BUS_ID)
        slot = CUDA.attribute(d, CUDA.DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
        bdf = string(domain; base = 16, pad = 4) * ":" *
            string(bus; base = 16, pad = 2) * ":" *
            string(slot; base = 16, pad = 2) * ".0"
        path = "/sys/bus/pci/devices/$bdf/numa_node"
        if isfile(path)
            node = tryparse(Int, strip(read(path, String)))
            node === nothing ? -1 : node
        else
            -1
        end
    catch
        # Topology introspection is a diagnostic convenience, never a test failure
        -1
    end
end

"""
    _far_device_pair() -> Union{Nothing, NamedTuple}

Pick the most topologically distant pair of CUDA devices available: the first
pair sitting on different NUMA nodes — a `SYS`-class link on a multi-socket host,
the class that has never been exercised by the suite — or `(0, NGPUS - 1)` when
NUMA affinity is unavailable or every GPU shares a node.

Returns `(devices, nodes, cross_numa)`, or `nothing` with fewer than two devices.
"""
function _far_device_pair()
    NGPUS >= 2 || return nothing
    nodes = [_numa_node(dev) for dev in 0:(NGPUS - 1)]
    for i in 1:NGPUS, j in (i + 1):NGPUS
        ni, nj = nodes[i], nodes[j]
        if ni >= 0 && nj >= 0 && ni != nj
            return (devices = (i - 1, j - 1), nodes = (ni, nj), cross_numa = true)
        end
    end
    return (
        devices = (0, NGPUS - 1),
        nodes = (nodes[1], nodes[end]),
        cross_numa = false,
    )
end

const FAR_PAIR = HAS_CUDA ? _far_device_pair() : nothing

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
    @info "Running GPU tests with $NGPUS device(s), sweeping $DEVICE_COUNTS"
    if FAR_PAIR !== nothing
        @info "Far device pair: $(FAR_PAIR.devices) " *
            "(NUMA $(FAR_PAIR.nodes), cross-socket: $(FAR_PAIR.cross_numa))"
    end
    include("test_ghost_exchange.jl")
    include("test_cross_socket.jl")
    include("test_vector.jl")
    include("test_broadcast.jl")
    include("test_matrix.jl")
    include("test_krylov.jl")
else
    @warn "CUDA not available or no GPUs detected, skipping GPU tests"
end
