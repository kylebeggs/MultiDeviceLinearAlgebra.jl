# Code defects in MultiDeviceLinearAlgebra and MatrixFreeOperators

**Date:** 2026-07-27 · **Reported by:** kbeggs
**Repos:** `~/dev/MultiDeviceLinearAlgebra` (MDLA), `~/dev/MatrixFreeOperators` (MFO)

---

> ## ⚠️ Correction (2026-07-27, after source-level review)
>
> **Defect 1 below — "No CUDA stream synchronization anywhere in MDLA", rated high — is
> incorrect, and the reasoning that depends on it (including the `_transfer_slab!` docstring
> criticism) is incorrect with it.** This document is retained as a record of the investigation;
> do not action Defect 1.
>
> The argument correctly observes that `Base.@sync` waits on Julia tasks rather than CUDA
> streams, and that CUDA.jl gives each task its own stream. It then stops one layer too early.
> CUDA.jl wraps every `CuArray`'s memory in a `Managed` struct that tracks which stream last
> touched it and **implicitly synchronizes on cross-stream or cross-device access**
> (`CUDA/src/memory.jl:571-575` for the stream case, `:550-552` for the device case; every
> `CuPtr` conversion routes through it at `CUDA/src/array.jl:464-467`). CUDA.jl's own docstring
> at `array.jl:496`: *"By default `CuArray`s are implicitly synchronized when they are accessed
> on different CUDA devices or streams."* MDLA never calls `enable_synchronization!`, so the
> default holds. Verified in CUDA.jl 5.9.6 and 5.11.3.
>
> The `_transfer_slab!` docstring's claim that the D→H copy synchronizes against the producing
> stream is likewise **correct** — `array.jl:606-616` does `context!(context(src)) do;
> synchronize(src)`.
>
> The genuinely useful findings here are the documentation and test-coverage items, not Defect 1.

---

## Why this document exists

These defects were found by code inspection during the `sasquatch` GPU peer-to-peer
investigation (see `GPU-P2P-incident-report.md`). **They are independent of that hardware
fault.** The host-level fix — `intel_iommu=off` plus a reboot — will not address any of them.
They were written down before the reboot so they do not get lost once the P2P symptoms clear.

All four claims below were verified against the source, not carried over from notes. The
verification command is given with each one.

There is an important interaction to be aware of: **defect 1 is a silent data race, and the
broken P2P has been masking it.** While every peer transfer was returning `nan`, no test could
have distinguished "race produced stale data" from "P2P produced garbage". After the reboot,
P2P will work and the race will remain — but it will now manifest as intermittent, load-dependent
wrong answers rather than uniform `nan`. That is a harder failure to diagnose, not an easier one.

---

## Defect 1 — No CUDA stream synchronization anywhere in MDLA (correctness, high)

**Severity: high.** Silent wrong results. No error raised.

MDLA contains no stream synchronization at all:

```bash
$ cd ~/dev/MultiDeviceLinearAlgebra && grep -rn 'synchronize' src/
$ echo $?
1        # no matches, anywhere in src/
```

### The race in `scatter!` (`src/ghost.jl:338-379`)

Phase 1 (`:346-355`) packs each device's send buffers:

```julia
@sync for d in 1:ndevices
    @async begin
        CUDA.device!(device_id(row_spec, d))
        for k in eachindex(ghost.neighbors[d])
            if !isempty(ghost.send_local_indices[d][k])
                ghost.send_buffers[d][k] .= x.partitions[d][ghost.send_indices_gpu[d][k]]   # :351
            end
        end
    end
end
```

Line 351 is a broadcast — it *launches* a kernel on device `d`'s stream and returns
immediately. Phase 2 (`:358-378`) then has a **different** device read that buffer:

```julia
copyto!(ghost.recv_buffers[d][k], ghost.send_buffers[nbr][k_in_nbr])   # :371 — reads nbr's buffer
```

The only barrier between them is `Base.@sync`, which waits for Julia **tasks** to finish, not
for CUDA **streams** to drain. Under CUDA.jl each Julia task carries its own task-local stream,
so when the Phase-1 `@async` body returns, its packing kernel may still be in flight. Device `d`
can therefore read `send_buffers[nbr]` before device `nbr` has finished writing it, and consume
stale or partially-written data.

Note precisely what is and is not ordered here: even if the line-371 peer copy is ordered on
device `d`'s own stream, that ordering says nothing about device `nbr`'s stream. Cross-stream,
cross-device ordering is exactly what is missing.

### The same race in `reduce!` (`src/ghost.jl:411-457`)

Structurally identical. Phase 1 (`:420-438`) packs ghost contributions into
`ghost.recv_buffers[d][k]` (`:431-434`); Phase 2 (`:441-455`) has a different device read them:

```julia
copyto!(ghost.send_buffers[d][k], ghost.recv_buffers[nbr][k_in_nbr])   # :447 — reads nbr's buffer
```

Same `Base.@sync`-only barrier, same defect.

### A second, separate exposure: neither function synchronizes before returning

Phase 2 of both functions launches asynchronous work and then returns without draining it.
`scatter!` returns the value of its Phase-2 `@sync for` (`:358`); `reduce!` returns `x` (`:456`).
In both cases the caller receives a handle to data that may still be being written. Any caller
that reads `local_x` or `x.partitions` without its own synchronization is racing, independent of
the inter-phase race above.

### Fix

Add `CUDA.synchronize()` as the last statement of each Phase-1 `@async` body — at
`src/ghost.jl:354` (scatter!) and `:437` (reduce!). This drains the task-local stream on the
current device before the task completes, so the enclosing `@sync` becomes a real barrier
across all devices.

Add the same call at the end of each Phase-2 `@async` body (`:377`, `:454`) so the functions do
not return with work in flight.

Verify the fix with a test that runs enough devices and enough data to actually expose the
window — a 2-device test on small buffers will pass either way, because the packing kernel
completes before the next device gets scheduled. This is why the existing suite never caught it
(see defect 3).

---

## Defect 2 — Stale documentation: `_allgather_x!` does not exist (documentation, low)

**Severity: low**, but actively misleading to whoever picks up the P2P workaround.

`MatrixFreeOperators/test/multigpu/README.md:69` directs the reader to fix broken P2P by
bouncing cross-device copies through the host, "as `_allgather_x!` already does":

```
The fix belongs upstream in MDLA
(bounce the two cross-device copies through the host, as `_allgather_x!` already
does) or in the host's IOMMU/ACS configuration.
```

There is no such function. It does not exist in MDLA at all, and within MFO it appears only in
this one README sentence:

```bash
$ cd ~/dev/MultiDeviceLinearAlgebra && grep -rn '_allgather_x' . ; echo $?
1                                              # no matches in MDLA

$ cd ~/dev/MatrixFreeOperators && grep -rn '_allgather_x' .
test/multigpu/README.md:69: ... as `_allgather_x!` already does ...    # the only hit
```

The practical consequence: anyone implementing the host-staging workaround will go looking for
an existing implementation to copy and will not find one. **The host-staging path must be
written from scratch, not reused.**

### Fix

Correct the sentence to say the host-staging path does not yet exist and would need to be
written, pointing at the two real cross-device copy sites — `src/ghost.jl:371` and `:447`.

### Standing caveat on that workaround

Host staging restores *correctness* under broken P2P but not performance — measured throughput
on this host was ~1 GB/s against an expected 20–25 GB/s. It is a stopgap for a machine that
cannot be rebooted, not a fix. With the IOMMU change going in, it should not be needed at all.

---

## Defect 3 — Test coverage never exercises the production device count (test gap, high)

**Severity: high.** This is the reason defect 1 has gone unnoticed.

MDLA defaults to using **every** visible GPU in production — 9 on this host:

```julia
src/vector.jl:57:  function MultiDeviceVector(v::Vector{T}; ndevices::Int = length(CUDA.devices()))
src/matrix.jl:22:      A::SparseMatrixCSC{Tv, Ti}; ndevices::Int = length(CUDA.devices())
```

But every test caps out far below that. In MDLA's `test/test_ghost_exchange.jl`, seven loops cap
at 4 and two more at 2:

| Line | Cap |
|---|---|
| 3, 45, 99, 156, 258, 281 | `for ndev in 1:min(NGPUS, 4)` |
| 202 | `ndev = min(NGPUS, 2)` |
| 320 | `compute_partition_ranges(20, min(NGPUS, 2))` |

MFO is capped at 2 — `test/mdla_gpu.jl:153`: `for nd in 1:min(NGPUS_MDLA, 2)`.

So nothing automated has ever run the 8- or 9-device topology that production actually uses.
Two distinct risks follow from that, and they compound:

1. **Scale-dependent bugs are invisible.** The defect-1 race widens as device count and neighbor
   count grow — more concurrent streams, more cross-device reads per phase, more scheduling
   slack for a Phase-2 read to overtake a Phase-1 write. A 2- or 4-device run is close to the
   best case for hiding it.
2. **Topology-dependent bugs are invisible.** On `sasquatch`, devices 0–3 all sit on NUMA node 0
   (see the topology matrix in `GPU-P2P-incident-report.md`). A 4-device test therefore never
   crosses a socket. The `SYS`-class GPU pairs — every 0–3 against 4–8 — have never been
   exercised by any test, at any point.

### Fix

Raise the caps to `NGPUS` where the tests are genuinely device-count agnostic, and add at least
one case that spans NUMA nodes (e.g. devices 0 and 4) so cross-socket paths are covered. Run the
full suite at 9 devices after the reboot.

Do this **after** the P2P fix is confirmed — running the wide suite on the current host would
only reproduce the IOMMU fault, and the 36-pair sweep is what wedged the driver.

---

## Defect 4 — Repeated fancy-index temporaries in `reduce!` (performance, low)

**Severity: low.** Correctness is unaffected; noted while reading the same code.

`src/ghost.jl:448-450`:

```julia
x.partitions[d][ghost.send_indices_gpu[d][k]] .= op.(
    x.partitions[d][ghost.send_indices_gpu[d][k]],
    ghost.send_buffers[d][k],
)
```

Indexing a `CuArray` with an index array allocates a new device array. The same gather is
written twice here, so each neighbor iteration allocates two temporaries where one would do —
and the read-modify-write could be a single fused kernel over the index set instead. The same
pattern appears at `:351`, where the allocation is inherent to the gather.

Worth folding into whatever change addresses defect 1, since it touches the same lines. Not
worth a standalone change.

---

## Summary

| # | Defect | Severity | Location |
|---|---|---|---|
| 1 | No CUDA stream sync; cross-device read/write race | **High** — silent wrong results | `MDLA src/ghost.jl:346-378`, `:420-455` |
| 2 | `_allgather_x!` referenced but does not exist | Low — misleading | `MFO test/multigpu/README.md:69` |
| 3 | Tests cap at 4 (MDLA) / 2 (MFO); production uses 9 | **High** — masks #1 | `MDLA test/test_ghost_exchange.jl`, `MFO test/mdla_gpu.jl:153` |
| 4 | Duplicated fancy-index temporaries | Low — performance | `MDLA src/ghost.jl:448-450` |

Recommended order: fix **1**, then **3** to prove 1 is actually fixed at 9 devices, then **2**
and **4** as cleanup. All of this is independent of the host reboot and can proceed on any
machine with working P2P.
