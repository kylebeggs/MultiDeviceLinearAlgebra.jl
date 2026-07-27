# Post-Fix Validation — `sasquatch` GPU P2P

**Date:** 2026-07-27 15:08–15:45 · **Status:** PASS. P2P correctness and performance both confirmed
healthy. Host is cleared for multi-GPU production work.

Follows the Part 5 checklist in `GPU-P2P-incident-report.md`, run after the `intel_iommu=off`
reboot.

## Host state after reboot

| Check | Before | After |
|---|---|---|
| Kernel | 6.17.0-35-generic | **7.0.0-28-generic** (also upgraded) |
| `/proc/cmdline` | no iommu flag | `... quiet splash intel_iommu=off vt.handoff=7` |
| `iommu_group/type` | `DMA-FQ` (all 9 GPUs) | **no `iommu_group` node at all** — IOMMU fully off |
| `nvidia-smi` | wedged, unresponsive | responsive, all 9 GPUs enumerated |
| Stuck `D`-state procs | 5+, unkillable | none |
| Driver | 535.309.01 | 535.309.01 (unchanged) |

## Step 2 — single pair 0↔1 (the decisive test)

```
cudaMemcpyPeer / cudaMemcpy between GPU0 and GPU1: 16.53GB/s
Test passed
```

Before: **1.07 GB/s, every element `nan`, FAIL**. A 15× bandwidth jump and clean verification.

## Step 3 — full 36-pair sweep

**36/36 pairs `Test passed`. Zero verification errors. No hangs, no driver instability.**
Raw output: `03-simpleP2P-all-pairs-postfix.txt`.

| Statistic | Value |
|---|---|
| Range | 14.39 – 17.84 GB/s |
| Slowest pair | 0↔4 (`SYS`, cross-socket) — 14.39 GB/s |
| Fastest pair | 4↔7 (`NODE`) — 17.84 GB/s |
| Same-switch (`PIX`) pairs | 0↔1 16.46, 2↔3 16.11, 4↔5 16.58, 6↔7 16.29, 6↔8 16.97, 7↔8 16.39 |

## Step 4 — NCCL all-reduce, all 9 GPUs

```
./nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 4 -g 9 -c 1 -n 20
```

**`#wrong = 0` on every row, both in-place and out-of-place. `Out of bounds values: 0 OK`.**
Peak bus bandwidth 15.38 GB/s at 128 MB; avg 4.26 GB/s across the size sweep.
Raw output: `04-nccl-all_reduce.txt`.

## Hardware health after the full sweep

0 uncorrected ECC on all 9 GPUs · 0 `Xid`/`NVRM` entries this boot · no processes in `D` state.

---

## Step 5 — PCIe ACS and Stage 3 bandwidth/latency matrices

The flat ~16–17 GB/s across topologies in step 3 initially looked like ACS forcing peer traffic to
the root complex. **It is not.** Both follow-up checks came back clean:

**ACS is disabled everywhere.** All 48 bridges report every bit clear
(`05-acs-status.txt`):

```
ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
```

**All 9 links are at full Gen4 x16** (`pcie.link.gen.current = 4`, `width.current = 16`).

**`p2pBandwidthLatencyTest` confirms P2P is genuinely engaged** (`06-p2pBandwidthLatencyTest.txt`).
The latency matrices are the decisive evidence — latency cannot be faked by a host-staged path:

| Metric | P2P Disabled | P2P Enabled |
|---|---|---|
| **Latency** | 13.3 – 21.6 µs | **1.65 – 2.70 µs** (~10× better) |
| Bidirectional BW | 21.4 – 30.5 GB/s | **32.5 – 51.2 GB/s** |
| Unidirectional BW | 17.6 – 21.2 GB/s | 16.3 – 25.8 GB/s (≈ parity) |

A ~10× latency drop and a clear bidirectional throughput gain only happen when transfers actually
traverse the peer path. The unidirectional-bandwidth parity is expected, not a defect: a single
streaming copy is bounded by the same Gen4 x16 link either way, and on this host the staged path
through fast host memory keeps up. The P2P advantage shows up in latency and in bidirectional
throughput, and it is clearly present.

**Conclusion: performance is healthy. No BIOS change and no further reboot is needed.**

Note that `simpleP2P`'s ~16 GB/s figure and this test's numbers measure different things —
`simpleP2P` ping-pongs with synchronization overhead included, so it reads lower. The Stage 3
matrices are the better performance reference. (The sample itself carries the disclaimer that CUDA
Samples are not intended for performance measurement.)

## Still open from the original report (unrelated to the host fix)

1. **MDLA has no CUDA stream synchronization** — `scatter!` (`src/ghost.jl:346-378`) and `reduce!`
   (`:420-455`) race Phase-1 packing kernels against Phase-2 peer reads. `Base.@sync` waits on Julia
   tasks, not CUDA streams. This bug survived the reboot and is now the *only* remaining source of
   silent wrong numbers. Fix: `CUDA.synchronize()` at the end of each Phase-1 `@async` body.
2. **Test coverage gap** — MDLA suites cap at `min(NGPUS, 4)`, MFO at 2. The 9-device topology is
   now known-good at the hardware level and can finally be exercised.
3. **Stale doc** — `MatrixFreeOperators/test/multigpu/README.md:69` references `_allgather_x!`,
   which does not exist in MDLA.
4. **Inventory discrepancy** — notes say 10 GPUs, `nvidia-smi` still enumerates 9 after a clean
   reboot. If the chassis holds 10 cards, one is off the bus.
