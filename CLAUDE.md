# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MultiDeviceLinearAlgebra.jl distributes dense vectors and sparse CSR matrices across multiple NVIDIA GPUs, exposing them through Julia's `AbstractArray` interface. It provides LinearAlgebra operations (dot, norm, axpy!, mul!), broadcasting, and Krylov.jl integration for iterative solvers — all parallelized across devices using `@sync`/`@async` with `CUDA.device!()` context switching.

## Common Commands

```bash
# Run all tests (requires CUDA GPUs; partition tests run on CPU)
julia --project -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project test/test_partition.jl

# Run benchmarks (set POISSON_NX env var for grid size, default 500)
POISSON_NX=200 julia --project scripts/bench_poisson.jl

# Format code (uses BlueStyle via .JuliaFormatter.toml)
julia -e 'using JuliaFormatter; format(".")'

# Build docs locally
julia --project=docs docs/make.jl
```

## Architecture

**Single module, no submodules.** All source files are `include()`d into `MultiDeviceLinearAlgebra` in dependency order.

### Core types
- `PartitionSpec{R,D}` — immutable metadata describing how indices are split across `ndevices` GPUs (ranges, total length, device ID mapping); supports manual construction from custom ranges via `PartitionSpec(ranges; devices=nothing)`
- `MultiDeviceVector{T} <: AbstractVector{T}` — partitioned dense vector; wraps `Vector{CuVector{T}}` (one per device) + `PartitionSpec`
- `MultiDeviceSparseMatrixCSR{Tv,Ti,GE} <: AbstractMatrix{Tv}` — row-partitioned sparse CSR matrix; each device holds its row block as `CuSparseMatrixCSR` with locally-remapped column indices; owns a `GhostExchange` for P2P halo communication
- `GhostExchange{Tv,V,VI}` — pre-computed ghost/halo communication topology and GPU buffers for P2P exchange between devices; can be constructed from a matrix (automatic ghost discovery) or from explicit per-device ghost index lists (for FEM assembly)

### Concurrency model
All multi-device operations follow the same pattern: `@sync` block with `@async` per device, each calling `CUDA.device!(device_id(spec, d))` before operating on that device's partition. SpMV (`mul!`) uses a ghost/halo P2P exchange — each device only fetches the off-partition column values it needs from neighboring devices, then performs local SpMV with remapped column indices.

### Ghost exchange (P2P halo communication)
At matrix construction time, each device's CSR column indices are analyzed to determine which off-partition (ghost) values are needed. Column indices are remapped to local numbering (`1:n_owned` for owned, `n_owned+1:n_owned+n_ghost` for ghosts). Before each SpMV, `scatter!` exchanges ghost values between devices via GPU staging buffers: pack → P2P transfer → assemble into `local_x = [owned | ghost]`. A `GhostExchange` can also be constructed independently from user-specified ghost indices (for FEM assembly), and `reduce!` sends ghost contributions back to owners with a user-supplied binary op.

### Key source files (in include order)
1. `partition.jl` — `PartitionSpec`, `compute_partition_ranges()`, `device_for_index()`, `device_id()`
2. `vector.jl` — `MultiDeviceVector` constructors and AbstractArray interface
3. `vector_linalg.jl` — `dot`, `norm`, `axpy!`, `axpby!`, `rmul!`, `lmul!`
4. `vector_broadcast.jl` — `MultiDeviceVectorStyle` broadcasting
5. `ghost.jl` — `GhostExchange`, ghost topology/map computation, column remapping, `scatter!`, `reduce!`
6. `matrix.jl` — `MultiDeviceSparseMatrixCSR` constructor and interface
7. `mul.jl` — sparse matrix-vector multiply with ghost exchange
8. `gather.jl` — device-to-host transfer for vectors and matrices (reverses column remapping)
9. `krylov_compat.jl` — Krylov.jl CG workspace support, `mdla_solve()`
10. `poisson.jl` — `poisson_matrix_2d()` test problem generator

## Testing

Uses standard `Test` module (`@testset`/`@test`/`@test_throws`). GPU tests in `runtests.jl` are gated on `CUDA.functional()` and `length(CUDA.devices()) >= 1`. CPU-only tests (`test_partition.jl`, `test_ghost.jl`, `test_poisson.jl`) always execute. GPU tests (`test_ghost_exchange.jl`, `test_vector.jl`, `test_broadcast.jl`, `test_matrix.jl`, `test_krylov.jl`) sweep over `1:min(NGPUS, 4)` device counts.

## Julia Style & Naming
- `snake_case` for functions/variables, `CamelCase` for types/modules, `SCREAMING_SNAKE_CASE` for constants
- `!` suffix for mutating functions
- No type piracy

## Performance
- Type stability; concrete types in struct fields
- `const` for globals or pass as function arguments
- `@views` for array slices; pre-allocate outputs; prefer in-place operations
- Column-major iteration order
- No `Vector{Any}` or abstract element types

## Type System & Dispatch
- Multiple dispatch over if/else type-checking
- Don't over-constrain argument types — duck typing unless dispatch requires it
- Parametric types for generic structs; abstract types for dispatch hierarchies

## Error Handling
- Specific exception types (`ArgumentError`, `DimensionMismatch`, `BoundsError`, etc.)
- `throw` for errors, not control flow

## Package Conventions
- Explicit `export` — only public API
- Triple-quote docstrings (`"""..."""`) above functions
- BlueStyle formatting (`.JuliaFormatter.toml`)
