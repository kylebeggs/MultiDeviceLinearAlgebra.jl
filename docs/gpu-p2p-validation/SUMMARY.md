# GPU↔GPU Communication Validation — 9× A30 host

**Date:** 2026-07-27 · **Status: RESOLVED — this document is a historical incident record.**

> **⚠️ Superseded. Do not read the verdict below as current state.**
>
> The fault described here was fixed the same day by the `intel_iommu=off` reboot prescribed under
> [Fix](#fix). See **[`POST-FIX-VALIDATION.md`](POST-FIX-VALIDATION.md)** for the confirmation run:
> 36/36 pairs passing at 14.39–17.84 GB/s, NCCL all-reduce healthy across all 9 GPUs.
>
> Re-confirmed 2026-07-29 while collecting device benchmarks for #26: `/proc/cmdline` carries
> `intel_iommu=off` on kernel 7.0.0-28-generic, and MDLA's own `_p2p_copy_ok` probe returns ✓ for
> every ordered pair it was run against. P2P on this host works.
>
> Everything below is kept as written on 2026-07-27, because the diagnostic trail is the useful
> part — if this signature ever recurs, this is how it was identified.

## Verdict (as of 2026-07-27 — no longer true, see banner)

**Peer-to-peer GPU communication is broken on this host, on every pair tested — including the
best-case topology.** This is not a cross-socket problem and not an application bug. It is a
host-level PCIe P2P fault.

## Evidence (NVIDIA `cuda-samples` v12.0 `simpleP2P`, the gold-standard correctness check)

| Pair | Topology | Peer bandwidth | Data verification | Result |
|---|---|---|---|---|
| 0 ↔ 1 | `PIX` — same PCIe switch, same socket | **1.07 GB/s** | all elements `nan` | **FAIL** |
| 0 ↔ 2 | `NODE` — same socket, across host bridges | **0.97 GB/s** | all elements `nan` | **FAIL** |
| 0 ↔ 3 | `NODE` | — | — | **wedged the driver** |

Expected same-switch P2P bandwidth on Gen4 x16 is ~20–25 GB/s. **~1 GB/s is not P2P at all** — it
is the signature of every transfer being bounced through host memory under IOMMU translation.

An earlier standalone run of the same test on pair 0↔1 also produced:

```
CUDA error at simpleP2P.cu:200 code=719(cudaErrorLaunchFailure) "cudaDeviceSynchronize()"
```

That is a kernel on GPU1 faulting while dereferencing a peer pointer into GPU0's memory. So both
P2P mechanisms are broken: the memcpy path silently returns garbage (`nan`), and the direct
kernel-load path faults outright.

Critically, `0↔1` is the *most favourable* pair on the box — same PCIe switch, same NUMA node.
Cross-socket UPI is therefore **not** the explanation; the fault is system-wide.

## Root cause

The IOMMU is enabled in translating mode:

```
0000:2b:00.0 -> DMA-FQ      (all 9 GPUs identical)
/proc/cmdline: BOOT_IMAGE=... ro quiet splash vt.handoff=7     # no iommu=pt, no intel_iommu=off
```

NVIDIA requires VT-d to be **disabled or in passthrough** for PCIe peer-to-peer. Under active
translation, peer writes carry addresses the target GPU never mapped. The observed combination —
`cudaDeviceCanAccessPeer` returning true for all 72 ordered pairs, ~1 GB/s "peer" bandwidth, `nan`
payloads, and a launch fault on direct peer access — is the textbook signature.

PCIe ACS on the upstream Broadcom/PLX switch ports is the other half of this signature and was not
readable without root. Worth confirming:

```
sudo lspci -vvv | grep -i -B12 'ACSCtl' | grep -E 'ACSCtl|^[0-9a-f]{2}:'
```

Hardware itself is healthy: 0 uncorrected ECC, 0 corrected aggregate, 0 remapped rows, no
remapping failures, no `Xid` entries, all 9 links at full Gen4 x16.

## Fix

The real fix is a kernel cmdline change plus reboot:

```
GRUB_CMDLINE_LINUX_DEFAULT="... intel_iommu=off"      # or iommu=pt
sudo update-grub && sudo reboot
```

Then re-run Stage 2 to confirm all 36 pairs pass. A code-side host-staging workaround in MDLA
(`src/ghost.jl:371`, `:447`) would restore *correctness* without a reboot, but at ~1 GB/s it
cannot restore performance — the workaround is a stopgap, not a fix.

## Secondary defect found by inspection (independent, still real)

`MultiDeviceLinearAlgebra` has **no CUDA stream synchronization anywhere** — `grep -rn synchronize
src/` returns nothing. In `scatter!` (`src/ghost.jl:346-378`) Phase 1 packs `send_buffers[d]` with
a kernel on device `d`'s stream, and Phase 2 has a *different* device peer-read that buffer. The
only barrier is `Base.@sync`, which waits for Julia **tasks**, not CUDA **streams**. `reduce!`
(`:420-455`) has the same structure.

This is a genuine race independent of the P2P fault, and it will still be there after the IOMMU is
fixed. Fix: `CUDA.synchronize()` at the end of each Phase-1 `@async` body.

**Still open as of 2026-07-29.** `grep -rn synchronize src/` now returns one hit, but it is
`_probe_p2p_copy` (`src/ghost.jl:178`), a construction-time probe — not the hot path. Phase 1 of
`scatter!` still only enqueues its gather kernels, and `@sync` still waits on Julia tasks rather
than CUDA streams, so the ordering described above is unchanged. Line numbers have moved since this
was written (`scatter!` is now ~`:502`, `reduce!` ~`:600`).

## Test coverage gap

MDLA's suites cap at `min(NGPUS, 4)` (`test/test_ghost_exchange.jl:3,45,99,156,258,281`) and MFO at
2 (`test/mdla_gpu.jl:153`). Nothing automated has ever exercised the 8- or 9-device topology, and
MDLA defaults to `ndevices = length(CUDA.devices())` = 9 in production (`src/vector.jl:57`,
`src/matrix.jl:22`).

## Stale doc to correct

`MatrixFreeOperators/test/multigpu/README.md:69` says the host-staging fix belongs upstream "as
`_allgather_x!` already does". **`_allgather_x!` does not exist in MDLA** — `grep -rn _allgather_x`
finds it only in that README. The host-staging path must be written, not reused.

## Stages not run *(at the time — since completed)*

Stage 3 (`p2pBandwidthLatencyTest`), Stage 4 (NCCL correctness), Stage 5 (Julia pair matrix +
missing-barrier test) were all skipped: the driver wedged, and the Stage 2 verdict is already
unambiguous. Binaries are built and ready under `../` for a re-run after the host is fixed.

**Update:** Stages 3 and 4 were run after the reboot and passed — see
[`POST-FIX-VALIDATION.md`](POST-FIX-VALIDATION.md) and `06-p2pBandwidthLatencyTest.txt`.

## Artifacts

| File | Contents |
|---|---|
| `00-baseline.txt` | full `nvidia-smi -q`, topology, all four `-p2p` matrices, IOMMU group types, PCIe tree |
| `01-nccl-tests-build.log` | nccl-tests build |
| `02-simpleP2P-all-pairs.txt` | the 3 completed pairs |
