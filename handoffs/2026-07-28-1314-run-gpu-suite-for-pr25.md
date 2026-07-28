---
slug: run-gpu-suite-for-pr25
created: 2026-07-28-1314
status: open
---

# Handoff: run the GPU test suite for PR #25 on the 9-GPU host

## Goal / why this matters

PR #25 (branch `test/device-count-caps-cross-numa`, closes #23) raises the test
device-count caps to every visible GPU and adds `test/test_cross_socket.jl`,
which exercises a NUMA-crossing device pair for the first time. **None of that
new coverage has ever executed.** CI is CPU-only, so its green checkmarks prove
only that the package precompiles, Runic is clean, and the CPU-only test files
still pass — the entire GPU block was skipped and `test_cross_socket.jl` was
never even parsed there.

Your job: run the suite on this 9×A30 host and report on the four points in
"What's left" below. **Yes, run the tests — that is the task.** (The global
"don't run tests unless told" rule is explicitly waived here.)

## Background & current state

MDLA defaults to every visible GPU in production (`src/vector.jl:57`,
`src/matrix.jl:22` — `ndevices::Int = length(CUDA.devices())`), but every GPU
testset capped at `min(NGPUS, 4)` or `min(NGPUS, 2)`. Since every spec came from
`compute_partition_ranges(n, ndev)` with the default `devices = 0:ndev-1`, a
4-device run only ever touched CUDA devices 0–3 — all on NUMA node 0. **No test
had ever crossed a socket.** The `SYS`-class pairs (0–3 × 4–8) are exactly the
ones the July 2026 IOMMU fault hit hardest and exactly the ones the
`_p2p_copy_ok` probe and its host-staging fallback (PR #22) exist to protect.

The PR is test-only — **no `src/` changes.** The plumbing already supported it:
`compute_partition_ranges(n; devices = [0, 4])` propagates end-to-end, and #22
deliberately allocated `host_buffers` unconditionally so tests could force the
fallback by flipping `p2p_ok` flags.

Verified on a Mac dev box before handing off: Runic clean, all ten test files
parse, CPU suite 203/203, CI green on Julia 1.10/1.11/1.12.

## Key files / locations

- `test/runtests.jl:11-17` — `DEVICE_COUNTS = unique(filter(<=(NGPUS), [1, 2, 4, NGPUS]))`
- `test/runtests.jl:19-46` — `_numa_node(dev)`, reads `/sys/bus/pci/devices/<BDF>/numa_node`
  with the BDF built from `CUDA.attribute` PCI domain/bus/device
- `test/runtests.jl:48-74` — `_far_device_pair()` / `FAR_PAIR`
- `test/test_cross_socket.jl` — the new coverage (273 lines, 7 testsets)
- `test/test_krylov.jl:64` — the `allequal(iter_counts)` assertion flagged below
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

## What's left / next steps

1. Check out and run:

   ```
   git fetch && git checkout test/device-count-caps-cross-numa
   julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()' 2>&1 | tee /tmp/mdla-9gpu.log
   ```

2. **Report the `@info` banner.** Expect:

   ```
   Running GPU tests with 9 device(s), sweeping [1, 2, 4, 9]
   Far device pair: (0, 4) (NUMA (0, 2), cross-socket: true)
   ```

   If `cross-socket: false`, **stop and debug `_numa_node`** — it means the new
   tests are silently running on an intra-socket pair and prove nothing. Cross-check
   against `nvidia-smi topo -m` (GPUs 0–3 → NUMA 0, GPUs 4–8 → NUMA 2, every 0–3 ×
   4–8 pair `SYS`).

3. **Report any failure in the `"Cross-socket device pair"` testset.** This is new
   coverage of both the P2P path and the forced host-staging fallback on a
   `SYS`-class pair — a failure here is a real finding about the host or the
   fallback, not a flaky test. Note which sub-testset and whether it was a P2P or
   a forced-host case.

4. **Watch `test_krylov.jl` "Iteration count consistency."** It asserts
   `allequal(iter_counts)` and now includes a 9-device point, which changes the
   dot-product reduction order (per-device partials summed on the host). If it
   fails by a single iteration, **do not drop the 9-device case** — relax the
   assertion to `maximum(iter_counts) - minimum(iter_counts) <= 1` and say so in
   your report. The invariant worth keeping is that partitioning doesn't change
   the algorithm, not that it's bit-identical.

5. **Report whether any `_p2p_copy_ok` warning fired.** On this host with
   `intel_iommu=off` there should be **none**. A warning after a clean 36/36
   `simpleP2P` sweep means the probe itself is suspect, not the hardware — see
   `handoffs/2026-07-27-1604-gpu-comms-validation-findings.md:278-303`.

6. Post the result as a comment on PR #25.

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
