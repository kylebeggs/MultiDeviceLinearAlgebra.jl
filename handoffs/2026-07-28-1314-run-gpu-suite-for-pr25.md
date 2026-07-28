---
slug: run-gpu-suite-for-pr25
created: 2026-07-28-1314
revised: 2026-07-28
status: done
---

# Handoff: run the GPU test suite for PR #25 on the 9-GPU host

## Outcome

**Done — the suite runs green on the 9×A30 host: 1291 passing, 0 failed, 0 errored.**
Banner confirmed `sweeping [1, 2, 4, 9]` and
`Far device pair: (0, 4) (NUMA (0, 2), cross-socket: true)`.
`Cross-socket device pair` passed 78/78 on both the P2P and forced-host-staging
paths. No `_p2p_copy_ok` warning fired. Krylov's `allequal(iter_counts)` held
across all four device counts, so it was left alone.

Getting there took two fixes, both on this branch:

- **`fd29972`** — production bug. `length(CUDA.devices())` returns an `Int32`, and
  Julia does not convert a typed keyword default before handing it to the body
  method, so `MultiDeviceVector(v)` and `MultiDeviceSparseMatrixCSR(A)` had *always*
  thrown `MethodError` on every CUDA host. Nothing caught it because every existing
  testset passed `ndevices` explicitly; this PR's new "Default ndevices" testsets are
  the first to call them bare. The same `Int32` reached `compute_partition_ranges`
  through `NGPUS` once the caps were raised.
- **`44e73ef`** — a stale assertion predating this PR: `test_vector.jl` still expected
  `similar(v).ghost_exchange === nothing`, but `similar` has propagated a
  `copy_exchange` since 43498af. It fails on `main` too.

The rest of this document is the original brief, kept for context.

## Goal / why this matters

PR #25 (branch `test/device-count-caps-cross-numa`, closes #23) raises the test
device-count caps to every visible GPU and adds `test/test_cross_socket.jl`,
which exercises a NUMA-crossing device pair for the first time. **None of that
new coverage had ever executed.** CI is CPU-only, so its green checkmarks prove
only that the package precompiles, Runic is clean, and the CPU-only test files
still pass — the entire GPU block was skipped and `test_cross_socket.jl` was
never even parsed there.

## Background & current state

MDLA defaults to every visible GPU in production (the `ndevices` keyword default on
the `MultiDeviceVector` / `MultiDeviceSparseMatrixCSR` convenience constructors in
`src/vector.jl` and `src/matrix.jl`), but every GPU testset capped at
`min(NGPUS, 4)` or `min(NGPUS, 2)`. Since every spec came from
`compute_partition_ranges(n, ndev)` with the default `devices = 0:ndev-1`, a
4-device run only ever touched CUDA devices 0–3 — all on NUMA node 0. **No test
had ever crossed a socket.** The `SYS`-class pairs (0–3 × 4–8) are exactly the
ones the July 2026 IOMMU fault hit hardest and exactly the ones the
`_p2p_copy_ok` probe and its host-staging fallback (PR #22) exist to protect.

The PR was written as test-only. That held until the run: the new default-`ndevices`
coverage exposed a genuine `src/` bug, fixed in `fd29972` (see Outcome). The rest of
the plumbing already supported the change:
`compute_partition_ranges(n; devices = [0, 4])` propagates end-to-end, and #22
deliberately allocated `host_buffers` unconditionally so tests could force the
fallback by flipping `p2p_ok` flags.

Verified on a Mac dev box before handing off: Runic clean, all ten test files
parse, CPU suite 203/203, CI green on Julia 1.10/1.11/1.12.

## Key files / locations

Cited by symbol rather than line — these shift with every edit to the file.

- `test/runtests.jl` — `NGPUS` (`Int`-converted, see Outcome), `DEVICE_COUNTS`,
  `_numa_node(dev)` (reads `/sys/bus/pci/devices/<BDF>/numa_node`, BDF built from
  `CUDA.attribute` PCI domain/bus/device), `_far_device_pair()` / `FAR_PAIR`
- `test/test_cross_socket.jl` — the new coverage (273 lines, 7 testsets)
- `test/test_krylov.jl` — the `allequal(iter_counts)` assertion in
  `"Iteration count consistency"`
- `src/ghost.jl:155-186` — `_p2p_copy_ok` and `_probe_p2p_copy`
- `src/ghost.jl:424-436` — `_transfer_slab!`, the P2P-vs-host-staging branch
- `docs/gpu-p2p-validation/GPU-P2P-incident-report.md:211-223` — topology matrix
- `handoffs/2026-07-27-1604-gpu-comms-validation-findings.md` — prior incident findings

## Decisions & conclusions

Already settled — **do not relitigate**:

- `DEVICE_COUNTS` is `[1, 2, 4, NGPUS]`, not a full `1:NGPUS` sweep. On this host
  that is `[1, 2, 4, 9]` — four iterations, same count as the old `1:4`, so the
  sweep itself costs no extra wall-clock while still hitting the production count.
- The far pair is discovered at runtime from NUMA affinity rather than hardcoded
  as `(0, 4)`. A hardcoded pair cannot run when `NGPUS < 5` and would silently
  stop being cross-socket on a differently-wired host.
- The two asymmetric-ghost-slab testsets in `test_ghost_exchange.jl` stay pinned
  at 2 devices on purpose — they are about one pair with unequal send/recv slab
  sizes, not about device count. Their cross-socket variants live in the new file.
- The fallback path is forced with `fill!(ghost.p2p_ok[d], false)`. No new
  production hook was added, and none is needed.

## Results (2026-07-28, 9×A30 host)

Julia 1.12.6, CUDA.jl 5.9.6, driver 580.173.02, `intel_iommu=off`.

| Testset | Result |
|---|---|
| Manual PartitionSpec construction | 14/14 |
| Ghost Exchange (CPU) | 107/107 |
| Poisson matrix construction | 5/5 |
| Poisson GPU solve | 12/12 |
| Ghost Exchange GPU | 497/497 |
| Convenience scatter!/reduce! | 176/176 |
| Host-staged fallback | 255/255 |
| **Cross-socket device pair** | **78/78** |
| MultiDeviceVector | 61/61 |
| MultiDeviceVector broadcasting | 5/5 |
| MultiDeviceSparseMatrixCSR | 52/52 |
| Krylov.jl CG integration | 20/20 |
| LinearSolve.jl integration | 4/4 |
| Iteration count consistency | 5/5 |

Against the five things the brief asked for:

1. **Banner** — `sweeping [1, 2, 4, 9]`, `Far device pair: (0, 4) (NUMA (0, 2),
   cross-socket: true)`. Matches `nvidia-smi topo -m`: GPUs 0–3 → NUMA 0
   (`0000:2b/2d/3a/3b:00.0`), GPUs 4–8 → NUMA 2 (`0000:ad/ae/bc/bd/be:00.0`), all
   0–3 × 4–8 pairs `SYS`. `_numa_node` resolves from real PCI addresses, so it stays
   correct regardless of CUDA enumeration order.
2. **Cross-socket testset** — 78/78, no failures, on both the P2P and the forced
   host-staging path including the SpMV round-trip.
3. **Krylov iteration consistency** — passed unchanged. `allequal(iter_counts)` held
   across `[1, 2, 4, 9]`; the 9-device reduction order did not shift the CG count, so
   the assertion was **not** relaxed.
4. **`_p2p_copy_ok` warnings** — none, in any run. The probe is not producing false
   positives here.
5. **Two real bugs found**, both fixed on this branch — see Outcome at the top.

Reported on PR #25 as
[comment 5109717327](https://github.com/kylebeggs/MultiDeviceLinearAlgebra.jl/pull/25#issuecomment-5109717327).

### Worth knowing for the next GPU run

- A top-level `@testset` that records a failure throws when it finishes, and that
  throw escapes `include()` — so one broken file aborts `runtests.jl` and every later
  file silently never runs. The first run died in `test_ghost_exchange.jl` and
  `test_cross_socket.jl` never executed. Check *which* files reported, not just the
  final tally.
- CI cannot catch anything in this class. With `HAS_CUDA` false, `NGPUS` takes the `0`
  literal branch of its ternary, so the GPU block is skipped and the `Int32` never
  appears. A green CI on this repo says nothing about the GPU path.
- The host is shared. Cap runs with `JULIA_CUDA_HARD_MEMORY_LIMIT` and skip
  `scripts/bench_poisson.jl` (500×500 default) unless benchmarking is the actual task.

## Gotchas / constraints

- **Do not add `CUDA.synchronize()` calls and do not touch the `_transfer_slab!`
  docstring.** The stream-race analysis that motivated those was refuted —
  CUDA.jl's `Managed` memory layer synchronizes implicitly on cross-stream /
  cross-device access. See
  `handoffs/2026-07-27-1604-gpu-comms-validation-findings.md:68-157`.
  `docs/gpu-p2p-validation/MDLA-MFO-code-defects.md` predates that retraction and
  still contains the refuted analysis — do not mine it without re-checking.
- The `_p2p_copy_ok` warning fires **once per ordered device pair per Julia
  session** (inside the `get!` closure at `src/ghost.jl:160-166`), and the cache
  at `:143` has no reset function. Don't write a `@test_logs` assertion against
  it — the result would depend on include order.
- Expect the suite to take longer than before. Not from the sweep, but because
  `test_vector.jl` and `test_broadcast.jl` now run every case at 9 devices instead
  of 4, and each additional device means another CUDA context.
- Format with **Runic, never JuliaFormatter**, if you change anything:
  `julia -m Runic --inplace .`
- No secrets or credentials are involved in this task.
