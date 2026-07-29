# MultiDeviceLinearAlgebra.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kylebeggs.github.io/MultiDeviceLinearAlgebra.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kylebeggs.github.io/MultiDeviceLinearAlgebra.jl/dev/)
[![Build Status](https://github.com/kylebeggs/MultiDeviceLinearAlgebra.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kylebeggs/MultiDeviceLinearAlgebra.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kylebeggs/MultiDeviceLinearAlgebra.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kylebeggs/MultiDeviceLinearAlgebra.jl)

Distribute dense vectors and sparse CSR matrices across multiple NVIDIA GPUs through Julia's `AbstractArray` interface. Operations from `LinearAlgebra`, broadcasting, and [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) iterative solvers run in parallel across devices — no manual device management required.

## Requirements

- Julia 1.10+
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) v5
- 1 or more NVIDIA GPUs (multi-GPU features require 2+)

### Host requirements for multi-GPU

Ghost exchange moves data between devices with PCIe peer-to-peer copies, so the **host** must be
configured to allow them:

- **VT-d / IOMMU must be off or in passthrough mode.** Boot with `intel_iommu=off` or `iommu=pt`
  (`amd_iommu=off` / `iommu=pt` on AMD). With the IOMMU in translating mode, peer transfers can
  silently return zeros or `nan` at a fraction of the expected bandwidth.
- **`CUDA.can_access_peer` is not a reliable health check.** On a misconfigured host it returns
  `true` for every pair while the transfers themselves corrupt data. Trust an actual data
  round-trip, not the capability query.

MultiDeviceLinearAlgebra probes each ordered device pair at `GhostExchange` construction with a
cached round-trip check. Pairs that fail fall back to host-staged transfers and emit a one-time
warning — correct results, reduced bandwidth. **If you see that warning, fix the host**; the
fallback is a safety net, not a supported operating mode.

## Installation

```julia
using Pkg
Pkg.add("MultiDeviceLinearAlgebra")
```

Or in the Pkg REPL (`]`):

```
add MultiDeviceLinearAlgebra
```

## Quick Start

Build a sparse system on CPU, distribute it across GPUs, solve with conjugate gradients, and gather the result back to the host:

```julia
using MultiDeviceLinearAlgebra
using LinearAlgebra, SparseArrays

# Assemble a 2D Poisson problem on CPU
nx = ny = 100
A_cpu = poisson_matrix_2d(nx, ny)
N = nx * ny
b_cpu = rand(N)

# Distribute across all available GPUs
A = MultiDeviceSparseMatrixCSR(A_cpu)
b = MultiDeviceVector(b_cpu)

# Solve with CG (Krylov.jl under the hood)
x, stats = mdla_solve(A, b; atol=1e-10, rtol=1e-10)

# Gather result back to CPU
x_cpu = gather(x)
```

## API Reference

### Partitioning

#### `compute_partition_ranges(n::Int, ndevices::Int; devices=nothing) → PartitionSpec`

Splits `n` indices as evenly as possible across `ndevices`. Returns a `PartitionSpec` containing the per-device index ranges, total length, and device count. Pass `devices` to assign specific 0-indexed CUDA device IDs.

#### `compute_partition_ranges(n::Int; devices::AbstractVector{Int}) → PartitionSpec`

Splits `n` indices across `length(devices)` partitions, using the given 0-indexed CUDA device IDs.

#### `PartitionSpec{R,D}`

```julia
struct PartitionSpec{R<:AbstractVector{UnitRange{Int}},D<:AbstractVector{Int}}
    ranges::R       # index range for each device
    len::Int        # total number of indices
    ndevices::Int   # number of devices
    devices::D      # 0-indexed CUDA device IDs
end
```

**Manual constructor:**

```julia
PartitionSpec(ranges::AbstractVector{<:UnitRange}; devices=nothing)
```

Build a `PartitionSpec` from explicit contiguous ranges. Ranges must start at 1 and be non-empty. When `devices` is `nothing`, device IDs default to `0:ndevices-1`.

#### `device_id(spec::PartitionSpec, d::Int) → Int`

Returns the 0-indexed CUDA device ID for partition `d`.

#### `device_for_index(spec::PartitionSpec, i::Int) → (device, local_index)`

Returns the 1-based device number and local index for global index `i`.

### Vectors

#### `MultiDeviceVector{T} <: AbstractVector{T}`

A dense vector partitioned across GPUs. Each device holds a `CuVector{T}` for its chunk.

**Constructors:**

```julia
# From a CPU vector — auto-partitions across all GPUs
MultiDeviceVector(v::Vector{T})

# From a CPU vector with explicit device count
MultiDeviceVector(v::Vector{T}; ndevices=2)

# From a CPU vector with a pre-computed partition
MultiDeviceVector(v::Vector{T}, spec::PartitionSpec)

# Uninitialized with a given partition
MultiDeviceVector{T}(undef, spec::PartitionSpec)
```

Supports `getindex`, `setindex!`, `similar`, `zero`, `fill!`, `copyto!`, and full broadcasting (`y .= α .* x .+ β .* z`).

### Matrices

#### `MultiDeviceSparseMatrixCSR{Tv,Ti,GE} <: AbstractMatrix{Tv}`

A row-partitioned sparse CSR matrix distributed across GPUs. Each device holds its block of rows as a `CuSparseMatrixCSR` with column indices remapped to local numbering. Ghost (off-partition) values are exchanged between devices via P2P transfers before each SpMV — only the needed values are communicated, not the entire vector.

**Constructors:**

```julia
# From a CPU SparseMatrixCSC — converts to CSR, computes ghost topology, and distributes
MultiDeviceSparseMatrixCSR(A::SparseMatrixCSC; ndevices=length(CUDA.devices()))

# From a CPU SparseMatrixCSC with an explicit partition
MultiDeviceSparseMatrixCSR(A::SparseMatrixCSC, row_spec::PartitionSpec)
```

### Operations

#### `gather(v::MultiDeviceVector{T}) → Vector{T}`

Transfers a distributed vector back to the CPU as a dense `Vector`.

#### `gather(A::MultiDeviceSparseMatrixCSR) → SparseMatrixCSC`

Transfers a distributed matrix back to the CPU as a `SparseMatrixCSC`.

#### `scatter!(x::MultiDeviceVector, ghost::GhostExchange, spec::PartitionSpec)`

Owner→ghost exchange: packs owned values into send buffers, performs P2P transfers, and assembles `local_x = [owned | ghost]` on each device. Called automatically by `mul!` before each SpMV.

#### `reduce!(x::MultiDeviceVector, ghost::GhostExchange, spec::PartitionSpec, op)`

Ghost→owner reduction: copies owned values from `ghost.local_x` into `x`, packs ghost contributions into buffers, transfers them to owner devices, and applies `op` element-wise. Used after FEM assembly to reduce shared DOF contributions back to owners.

#### `GhostExchange(ghost_global_indices, spec::PartitionSpec, ::Type{Tv})`

Construct a `GhostExchange` from user-specified per-device ghost index lists, independent of any matrix. Each `ghost_global_indices[d]` is a sorted vector of global indices that device `d` needs as ghosts.

#### `mdla_solve(A, b; kwargs...) → (x, stats)`

Solves `Ax = b` using Krylov.jl's conjugate gradient method. All keyword arguments are forwarded to `Krylov.cg`. Returns the solution vector `x` (as a `MultiDeviceVector`) and solver statistics.

#### `poisson_matrix_2d(nx, ny; T=Float64) → SparseMatrixCSC`

Generates the standard 5-point finite-difference Laplacian on an `nx × ny` grid with Dirichlet boundary conditions. Useful for testing and benchmarking.

## Supported Operations

| Category | Operations |
|---|---|
| **LinearAlgebra** | `dot`, `norm`, `axpy!`, `axpby!`, `rmul!`, `lmul!`, `mul!` |
| **SpMV** | `mul!(y, A, x)` and `mul!(y, A, x, α, β)` — sparse matrix-vector multiply with P2P ghost exchange |
| **Base** | `fill!`, `copyto!`, `similar`, `zero`, `getindex`, `setindex!` |
| **Broadcasting** | Full element-wise broadcasting (e.g., `y .= α .* x .+ β .* z`) |

## Krylov.jl Integration

`MultiDeviceVector` and `MultiDeviceSparseMatrixCSR` work directly with Krylov.jl solvers. A `CgWorkspace` constructor is provided for pre-allocated CG workspaces:

```julia
using Krylov

# Using the convenience wrapper
x, stats = mdla_solve(A, b; atol=1e-12, rtol=1e-12)

# Or calling Krylov.cg directly
x, stats = Krylov.cg(A, b; atol=1e-12, rtol=1e-12)
```

## Benchmarking

Benchmarks come in two tiers, split by what the hardware allows.

### `benchmark/gpu.jl` — multi-GPU, run by hand

The real numbers. Sweeps device counts and reports, with a speedup column for each:

- the fused `_gather!` / `_scatter_apply!` kernels against the broadcasts they replaced,
  including device allocation per call;
- `scatter!` / `reduce!` on a real Poisson halo;
- SpMV `mul!`;
- the CG solve, with iteration count, VRAM per device, and relative residual.

Every section asserts parity against a CPU or broadcast reference before timing, so a run
doubles as a smoke test on hardware CI cannot reach. The device inventory at the top prints the
`_p2p_copy_ok` probe matrix — check it first if numbers look wrong, since a host with broken P2P
silently degrades every transfer to host staging.

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

# Default 500×500 grid
julia --project=benchmark benchmark/gpu.jl

# Larger grid, more solve repetitions, explicit device sweep
POISSON_NX=1500 BENCH_NRUNS=10 BENCH_NDEVICES=1,2,4,8 julia --project=benchmark benchmark/gpu.jl
```

At the 500² default the problem is small enough that halo communication dominates and scaling
is flat to negative; push `POISSON_NX` to 1500–2000 before drawing strong-scaling conclusions.

### `benchmark/benchmarks.jl` — CPU, automatic on every PR

An [AirspeedVelocity](https://github.com/MilesCranmer/AirspeedVelocity.jl) suite that
`.github/workflows/Benchmark.yml` runs against both revisions of a pull request and posts as a
ratio table. GitHub-hosted runners have no GPU, so it covers the host-side construction path
only — matrix assembly, the CSC→CSR transpose, ghost discovery and topology, column remapping.
That path is O(nnz) and worth guarding, but a clean table says nothing about device performance.

```bash
julia --project=benchmark -e 'include("benchmark/benchmarks.jl"); run(SUITE)'
```

Anything added there must also exist on a pull request's base branch, since the same file is run
against both revisions.

### `scripts/check_poisson.jl` — correctness, not timing

Verifies the distributed solve against the manufactured exact solution `u = sin(πx)sin(πy)`.
