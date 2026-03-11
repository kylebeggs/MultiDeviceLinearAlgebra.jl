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
- `PartitionSpec` — immutable metadata describing how indices are split across `ndevices` GPUs (ranges, total length)
- `MultiDeviceVector{T} <: AbstractVector{T}` — partitioned dense vector; wraps `Vector{CuVector{T}}` (one per device) + `PartitionSpec`
- `MultiDeviceSparseMatrixCSR{Tv,Ti} <: AbstractMatrix{Tv}` — row-partitioned sparse CSR matrix; each device holds its row block as `CuSparseMatrixCSR`

### Concurrency model
All multi-device operations follow the same pattern: `@sync` block with `@async` per device, each calling `CUDA.device!(d-1)` (0-indexed) before operating on that device's partition. SpMV (`mul!`) uses an allgather step — each device gets a full copy of `x` before computing its partition of `y = A*x`.

### Key source files (in include order)
1. `partition.jl` — `PartitionSpec`, `compute_partition_ranges()`, `device_for_index()`
2. `vector.jl` — `MultiDeviceVector` constructors and AbstractArray interface
3. `vector_linalg.jl` — `dot`, `norm`, `axpy!`, `axpby!`, `rmul!`, `lmul!`
4. `vector_broadcast.jl` — `MultiDeviceVectorStyle` broadcasting
5. `matrix.jl` — `MultiDeviceSparseMatrixCSR` constructor and interface
6. `mul.jl` — sparse matrix-vector multiply with allgather
7. `gather.jl` — device-to-host transfer for vectors and matrices
8. `krylov_compat.jl` — Krylov.jl CG workspace support, `mdla_solve()`
9. `poisson.jl` — `poisson_matrix_2d()` test problem generator

## Testing

Uses standard `Test` module (`@testset`/`@test`/`@test_throws`). GPU tests in `runtests.jl` are gated on `CUDA.functional()` and `length(CUDA.devices()) >= 2`. The partition tests (`test_partition.jl`) run on CPU only and can always execute.

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
