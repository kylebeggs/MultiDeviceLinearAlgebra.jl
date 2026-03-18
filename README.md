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

#### `consistent!(x::MultiDeviceVector, ghost::GhostExchange, row_spec::PartitionSpec)`

Exchanges ghost values between devices: packs owned values into send buffers, performs P2P transfers, and assembles `local_x = [owned | ghost]` on each device. Called automatically by `mul!` before each SpMV.

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

A Poisson benchmark script is included at `scripts/bench_poisson.jl`. It assembles a 2D Poisson system, distributes it across 1 to N GPUs, and reports upload, warmup, and solve times.

```bash
# Default 500×500 grid
julia --project scripts/bench_poisson.jl

# Custom grid size
POISSON_NX=200 julia --project scripts/bench_poisson.jl

# Custom number of timed runs (default 5)
BENCH_NRUNS=10 julia --project scripts/bench_poisson.jl
```

The script sweeps over device counts (1, 2, …, N) and verifies the solution against the manufactured exact solution `u = sin(πx)sin(πy)`.
