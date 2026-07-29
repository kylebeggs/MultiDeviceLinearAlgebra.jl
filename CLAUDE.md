# MultiDeviceLinearAlgebra.jl

Distributes dense vectors and sparse CSR matrices across multiple NVIDIA GPUs behind Julia's
`AbstractArray` interface — LinearAlgebra ops, broadcasting, and Krylov.jl integration, parallelized
with `@sync`/`@async` plus `CUDA.device!()` context switching.

## Key files (single module — `include()` order *is* dependency order)
`src/partition.jl` → `src/vector.jl` → `src/vector_linalg.jl` → `src/vector_broadcast.jl` →
`src/ghost.jl` → `src/matrix.jl` → `src/mul.jl` → `src/gather.jl` → `src/krylov_compat.jl` →
`src/poisson.jl`

Ghost/halo P2P exchange lives in `src/ghost.jl`; SpMV in `src/mul.jl`.

## Gotchas
- **Format with Runic, never JuliaFormatter** — CI enforces Runic via `fredrikekre/runic-action`:
  ```
  julia -m Runic --inplace .
  ```
- **Some hosts silently corrupt direct GPU-to-GPU P2P copies** (broken IOMMU/ACS). Construction runs a cached data round-trip probe per ordered device pair (`_p2p_copy_ok`); failing pairs fall back to pre-allocated host staging buffers with a one-time warning. If you hit unexplained numerical garbage on a new machine, check that probe before suspecting the algorithm.
- Matrix column indices are **remapped to local numbering** at construction (`1:n_owned` owned, then ghosts); `src/gather.jl` reverses this on the way back to the host. Raw device column indices are not global indices.
- No submodules — adding an `include` in the wrong position breaks the load order.
- GPU tests are gated on `CUDA.functional()` and sweep `1:min(NGPUS, 4)` device counts. The CPU-only tests (`test/test_partition.jl`, `test/test_ghost.jl`, `test/test_poisson.jl`) always run, so a green suite on a CPU box proves very little.
- **Benchmarks are two-tier, and the CI tier is CPU-only.** `benchmark/benchmarks.jl` is the
  AirspeedVelocity `SUITE` that `.github/workflows/Benchmark.yml` runs on every PR
  (`ubuntu-latest`, no GPU) — it covers the host-side construction path only, so a clean ratio
  table proves nothing about device performance. `benchmark/gpu.jl` holds the real multi-GPU
  numbers and is run by hand on the GPU host:
  `POISSON_NX=200 julia --project=benchmark benchmark/gpu.jl` (also `BENCH_NRUNS`,
  `BENCH_NDEVICES`). Anything added to `benchmarks.jl` must exist on a PR's *base* branch too —
  `benchpkg` runs the same file against both revisions.
- Indexed device work goes through the fused `_gather!` / `_scatter_apply!` kernels in
  `src/ghost.jl`, not broadcasts. `buf .= x[idx]` looks fused but is not: `getindex(::CuVector,
  ::CuVector)` is evaluated eagerly and materializes a temporary per occurrence (issue #24).
