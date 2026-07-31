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

#### `MultiDeviceSparseMatrixCSR{Tv,Ti,GE,VP,P,PC} <: AbstractMatrix{Tv}`

A row-partitioned sparse CSR matrix distributed across GPUs. Each device holds its block of rows as a `CuSparseMatrixCSR` with column indices remapped to local numbering. Ghost (off-partition) values are exchanged between devices via P2P transfers before each SpMV — only the needed values are communicated, not the entire vector.

Rows and columns carry **independent** `PartitionSpec`s (`row_spec` and `col_spec`), so the matrix may be rectangular — a prolongation operator maps a coarse space to a fine one, and the two spaces are partitioned separately. For a square matrix built from a single spec, `A.col_spec === A.row_spec`.

Device-resident index arrays are always `Int32`, whatever index type the host `SparseMatrixCSC` used, because cuSPARSE's sparse-sparse routines accept only 32-bit indices. `gather` still returns host-native `Int` indices.

**Constructors:**

```julia
# From a CPU SparseMatrixCSC — converts to CSR, computes ghost topology, and distributes
MultiDeviceSparseMatrixCSR(A::SparseMatrixCSC; ndevices=length(CUDA.devices()))

# From a CPU SparseMatrixCSC with an explicit partition (square matrices only)
MultiDeviceSparseMatrixCSR(A::SparseMatrixCSC, row_spec::PartitionSpec)

# Rectangular: rows and columns partitioned independently
MultiDeviceSparseMatrixCSR(A::SparseMatrixCSC, row_spec::PartitionSpec, col_spec::PartitionSpec)
```

Both specs must span the same devices in the same order. An `x` passed to `mul!` must be partitioned like `A.col_spec` and `y` like `A.row_spec` — matching lengths alone are not accepted, since a differently-split vector of the right length would silently read the wrong entries.

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

Generates the standard 5-point finite-difference Laplacian on an `nx × ny` grid with Dirichlet boundary conditions. Grid point `(i, j)` is unknown `(j - 1) * nx + i`. Useful for testing and benchmarking.

#### `prolongation_matrix_2d(nx, ny; bx=2, by=2, T=Float64) → SparseMatrixCSC`

Piecewise-constant (injection) prolongation for the same `nx × ny` grid: fine point `(i, j)` belongs to the aggregate covering the `bx × by` block it falls in, with unit weight. The result is `nx*ny × nxc*nyc` where `nxc = cld(nx, bx)` and `nyc = cld(ny, by)`, so aggregates along the far edges may be partial.

Exactly one nonzero per row, which makes `P' * A * P` the aggregate-summed Galerkin operator — a two-level coarse-grid fixture with no algebraic-multigrid dependency. The coarse space is numbered column-major to match the fine one, which keeps a contiguous block of fine unknowns mapping to a contiguous block of coarse ones and the distributed halo a surface rather than an all-to-all.

## Supported Operations

| Category | Operations |
|---|---|
| **LinearAlgebra** | `dot`, `norm`, `axpy!`, `axpby!`, `rmul!`, `lmul!`, `mul!` |
| **SpMV** | `mul!(y, A, x)` and `mul!(y, A, x, α, β)` — sparse matrix-vector multiply with P2P ghost exchange, square or rectangular |
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

## Reproducibility across device counts

**Results are not bitwise reproducible across `ndevices`, and cannot be made so from Julia.**
The halo exchange itself is exact — with one DOF per GPU and integer-valued entries the SpMV
matches the CPU product bit for bit on every device count, and repeated calls are bit-identical
(`test/test_exact_exchange.jl`). What changes is the *order* the floating-point work is done in:

- **cuSPARSE** schedules CSR row blocks according to the local row count, so each partitioning
  accumulates a row's contributions in a different order.
- **CUBLAS** picks a reduction tree shape from the partition length, so per-device `dot`/`norm`
  partials are formed differently.

Neither is reachable through CUDA.jl. Measured at 1000² Poisson against the single-device
result: the scalar reductions agree to 0–1 ULP, while SpMV differs at the level of
`‖Δy‖/‖y‖ ≈ 6e-17` — machine epsilon, but not zero. So the top-level `sum` over per-device
partials in `src/vector_linalg.jl` is *not* where the difference enters, and making it
fixed-order would not buy reproducibility.

Two consequences worth knowing:

1. **Pick a tolerance above the attainable floor.** For a residual-based stop the floor is
   roughly `ε‖A‖‖x‖`. Asking for less means the solver can never converge on the true residual
   and stops only when its recursively updated residual drifts under the threshold — hundreds of
   iterations later, on a path set entirely by rounding, and therefore at a count that moves with
   the device count. `benchmark/gpu.jl` prints the floor next to the residual for this reason.
2. **Compare `ms/iter`, not wall-clock, across device counts.** If the iteration count differs,
   wall-clock conflates solver throughput with the number of iterations and can hide a real
   per-iteration speedup entirely.

On a well-posed problem the iteration count has been invariant in practice — identical across
1, 2, 4 and 9 devices at both 1000² and 2000² Poisson, to the same true residual — but that is
an observation about one host's CUBLAS/cuSPARSE, not a guarantee. Do not assert it as an
invariant, and do not read an ill-posed problem's spread as wasted work.

## Benchmarking

Benchmarks come in two tiers, split by what the hardware allows.

### `benchmark/gpu.jl` — multi-GPU, run by hand

The real numbers. Sweeps device counts and reports, with a speedup column for each:

- the fused `_gather!` / `_scatter_apply!` kernels against the broadcasts they replaced,
  including device allocation per call;
- `scatter!` / `reduce!` on a real Poisson halo;
- SpMV `mul!`;
- the CG solve, with iteration count, `ms/iter`, VRAM per device, and the relative residual
  printed next to the attainable residual floor.

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

### `scripts/diagnose_partition_sensitivity.jl` — why a result moved with the device count

Run this when a number changes with `ndevices` and it is not obvious whether that is a bug or
rounding. It separates the two, in that order:

- **§1/§2 — is communication sound?** One DOF per GPU with integer-valued entries makes the SpMV
  arithmetic exact, so the comparison is bitwise and any mismatch is communication with no
  rounding to hide behind. Run against a tridiagonal pattern (neighbours only) and a dense one
  (every ordered device pair, the only case here that crosses the socket). §2 then repeats the
  SpMV back-to-back with no host synchronization, the access pattern CG produces. **The script
  exits nonzero if either fails** — everything below is meaningless until they pass.
- **§3–§5 — where does the difference enter?** ULP distances for `dot`, `norm` and SpMV against
  the single-device result; whether the problem is well posed for the requested tolerance; and
  iteration counts across device counts for an ill-posed, a well-posed, and an exactly-solvable
  system.

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
DIAG_NX=1000 julia --project=benchmark scripts/diagnose_partition_sensitivity.jl
```

Knobs: `DIAG_NX`, `DIAG_NDEVICES`, `DIAG_REPEATS`, `DIAG_NRUNS` (drop to 1 to roughly halve §5).
