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
- **Benchmarks are two-tier, and the CPU tier cannot see device changes.**
  `benchmark/benchmarks.jl` is the AirspeedVelocity `SUITE` that `.github/workflows/Benchmark.yml`
  runs on every PR (`ubuntu-latest`, no GPU) — it covers the host-side construction path only, so a
  clean ratio table proves nothing about device performance. **A device-only change will show 1.00
  across that whole table; that is the correct output, not a null result.** Anything added to
  `benchmarks.jl` must exist on a PR's *base* branch too — `benchpkg` runs the same file against
  both revisions.
- **`benchmark/gpu.jl` holds the real multi-GPU numbers.** Run by
  `.github/workflows/BenchmarkGPU.yml` on the self-hosted runner, and by hand:
  `POISSON_NX=1000 julia --project=benchmark benchmark/gpu.jl` (also `BENCH_NRUNS`,
  `BENCH_NDEVICES`). Its §2 (`bench_indexed_kernels`) is single-device with no transfers and is the
  section to read for kernel-level changes; §3–§5 are communication-bound below ~1500² and scale
  flat-to-negative there.
- **The GPU host is shared with other researchers — never assume you own it.** Anything touching a
  device goes through `benchmark/gpu_preflight.sh` first, which samples utilisation, memory and
  resident compute processes over a window before claiming a subset, and
  `benchmark/gpu_watchdog.sh`, which flags a run that got contended. Pin with
  `CUDA_VISIBLE_DEVICES`, cap with `JULIA_CUDA_HARD_MEMORY_LIMIT`, and stay niced: MDLA otherwise
  defaults to *every* visible device and sizes its pool against whatever is free. **Busy means
  wait, never barge in** — `--wait N` re-probes without holding anything rather than taking a
  device someone is on, and `--wait-hook` lets a caller abandon a wait that stopped being worth
  finishing. `--prefer-far` inverts the selection to the *worst*-connected subset, which is what
  correctness runs want: handed a well-connected pair, `_far_device_pair()` reports
  `cross_numa = false` and `test/test_cross_socket.jl` silently degrades to re-testing an adjacent
  pair. See `docs/gpu-ci-runner.md`.
- The shell tooling in `benchmark/` needs **bash 4+** (`mapfile`, associative arrays), so none of it
  runs on a stock Mac — `/bin/bash` there is 3.2. `bash -n` is the most a dev box can check;
  `benchmark/test_gpu_preflight.sh` (mocked `nvidia-smi`, no real devices) runs on the GPU host.
- Indexed device work goes through the fused `_gather!` / `_scatter_apply!` kernels in
  `src/ghost.jl`, not broadcasts. `buf .= x[idx]` looks fused but is not: `getindex(::CuVector,
  ::CuVector)` is evaluated eagerly and materializes a temporary per occurrence (issue #24).
- **Results are not bitwise reproducible across device counts, and no `src/` change can make
  them so.** cuSPARSE's CSR row-block schedule depends on the local row count and CUBLAS's
  reduction tree on the partition length; neither is reachable from Julia. Measured: the scalar
  reductions agree to 0–1 ULP while SpMV differs at `‖Δy‖/‖y‖ ≈ 6e-17`, so fixing the order of
  `sum(partial)` in `src/vector_linalg.jl` addresses a mechanism that contributes nothing. Before
  treating an `ndevices`-dependent number as a bug, run
  `scripts/diagnose_partition_sensitivity.jl` — §1/§2 settle whether communication is sound
  (exit nonzero if not) and §3–§5 separate rounding from conditioning. See the README's
  "Reproducibility across device counts".
