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
- Benchmark grid size comes from an env var: `POISSON_NX=200 julia --project scripts/bench_poisson.jl` (default 500).
