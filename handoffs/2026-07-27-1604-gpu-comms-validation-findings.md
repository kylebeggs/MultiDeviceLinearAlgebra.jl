---
slug: gpu-comms-validation-findings
created: 2026-07-27-1604
revised: 2026-07-27
status: open
---

# Handoff: close out the sasquatch P2P investigation (docs, hygiene, test coverage)

> **Revised 2026-07-27 after a source-level fact-check.** The original version of this doc led
> with a "genuine CUDA stream race" (Findings 1–3, two rated HIGH) and prescribed inserting four
> `CUDA.synchronize()` calls plus a docstring rewrite. **That analysis was wrong** — CUDA.jl
> already provides the ordering it claimed was missing. The refutation is preserved below in
> full, with citations, because the argument was plausible enough that the next reader will
> likely re-derive it. Everything else in the original doc checked out and survives.

## Goal / why this matters

The `sasquatch` GPU host had broken PCIe peer-to-peer; that is **fixed and re-validated**. What
remains is cleanup: stale documentation in two repos, a genuine test-coverage gap, and 875 MB of
untracked-and-unignored vendored tooling sitting in an iCloud-synced repo.

There is **no known correctness bug in MDLA**. The numerical garbage seen during the incident was
entirely the host IOMMU fault.

Two repos are in scope:
- MDLA — `~/dev/MultiDeviceLinearAlgebra` (this repo), GitHub `kylebeggs/MultiDeviceLinearAlgebra.jl`
- MFO — `~/dev/MatrixFreeOperators`, GitHub `RallypointOne/MatrixFreeOperators.jl`

## Background & current state

**Host fault (resolved, no action needed).** All 9× A30 GPUs reported
`iommu_group/type = DMA-FQ` with no `iommu=pt` / `intel_iommu=off` on the kernel cmdline.
NVIDIA requires VT-d off or in passthrough for PCIe P2P; under translation, peer transfers
silently returned `nan`/zeros at ~1 GB/s while `cudaDeviceCanAccessPeer` reported `true` for
all 72 ordered pairs. Fixed with `intel_iommu=off` + reboot. Post-fix: 36/36 `simpleP2P` pairs
pass at 14.4–17.8 GB/s, NCCL all-reduce `#wrong = 0` on all 9 GPUs, ACS clear everywhere, all
links Gen4 x16.

**Reports** (currently untracked, see step 4): `gpu-comms-validation/results/` —
`GPU-P2P-incident-report.md`, `POST-FIX-VALIDATION.md`, `SUMMARY.md`,
`MDLA-MFO-code-defects.md`, plus raw logs. Note `MDLA-MFO-code-defects.md` predates `53d8d07`
and contains the same refuted race analysis — do not mine it for work without re-checking.

**Repo state.** Branch `fix/ghost-p2p-fallback`, open as **PR #22**. Related: MDLA issue **#21**
(open, the silent-zeros report), MFO issue **#30** (open, CI for `test/multigpu`).

The branch is now **five commits ahead of `main` and four ahead of `origin` (unpushed)**:

| Commit | |
|---|---|
| `53d8d07` | `fix(ghost): probe P2P per device pair, host-stage transfers where broken` — the only one PR #22 currently shows |
| `f41d1b5` | `docs(claude): rewrite CLAUDE.md to gotchas-only` — committed by Kyle mid-session |
| `d7007df` | `docs(readme): document host P2P/IOMMU requirement` |
| `295299d` | `chore(repo): track GPU P2P validation reports, ignore vendored CUDA tooling` |
| `1b81f54` | `docs(handoff): retract refuted stream-race analysis` (this document) |

Pushing the branch will widen PR #22 from "P2P probe" to "probe + docs + hygiene". If you want
PR #22 to stay tight, split the last three onto their own branch off `main` before pushing.

**Decision already made by Kyle: keep PR #22.** The probe + host-staging fallback stays as
defense-in-depth even though the host fault is fixed — the probe's `@warn` converts a
silent-wrong-answer class into a loud one, and this was a demonstrated failure, not a
hypothetical. Do not revert it.

---

## Refuted: the "CUDA stream race" (originally Findings 1, 2, 3, 8)

**Do not implement the four `CUDA.synchronize()` insertions. Do not rewrite the
`_transfer_slab!` docstring. Do not "correct" issue #21's synchronization conclusion.**

### The argument that was made

`scatter!` and `reduce!` are two-phase: Phase 1 packs buffers on each device, Phase 2 has a
*different* device read those buffers. The packing at `src/ghost.jl:459` is a broadcast — it
launches a kernel on the task-local stream and returns immediately. The only barrier is
`Base.@sync`, which waits for Julia **tasks**, not CUDA **streams**, and CUDA.jl gives each task
its own stream. Therefore device `d` can read `send_buffers[nbr]` before device `nbr` finished
writing it. `src/mul.jl` compounds it by running the SpMV in a fresh task ⇒ fresh stream,
unordered against the stream that assembled `local_x`.

Both premises are true. The conclusion does not follow.

### Why it's wrong

The argument stops one layer above where the ordering actually lives. CUDA.jl wraps every
`CuArray`'s memory in a `Managed` struct that tracks *which stream last touched it* and
synchronizes on cross-stream or cross-device access.

`~/.julia/packages/CUDA/*/src/memory.jl:571-575` — the cross-stream case:

```julia
# accessing memory on another stream: ensure the data is ready and take ownership
if managed.stream != state.stream
  maybe_synchronize(managed)      # → synchronize(managed.stream) when dirty
  managed.stream = state.stream
end
```

`memory.jl:550-552` does the same for cross-device access. Every device-pointer conversion of a
`CuArray` routes through it — `array.jl:464-467`, the comment says so outright:

```julia
# defer the conversion to Managed, where we handle memory consistency
Base.unsafe_convert(typ::Type{CuPtr{T}}, x::CuArray{T}) where {T} =
  convert(typ, x.data[]) + x.offset * Base.elsize(x)
```

CUDA.jl's docstring for the opt-out (`array.jl:496-500`) states the guarantee plainly:

> By default `CuArray`s are implicitly synchronized when they are accessed on different CUDA
> devices or streams.

**MDLA never calls `enable_synchronization!`** (grep of `src/` and `test/`: zero hits), so the
default applies everywhere. Verified identical in CUDA.jl **5.9.6** (the `Manifest.toml` pin) and
**5.11.3** (what issue #21 reports the affected host ran).

Walking the exact scenario: Phase 1 on device `nbr` writes `send_buffers[nbr][k]`, leaving
`managed.stream` = that task's stream and `dirty = true`. Phase 2's task converts a pointer to
that same buffer from a different stream → `maybe_synchronize` → blocks until the packing kernel
retires. Same mechanism covers the `src/mul.jl` SpMV reading `local_x[d]`.

### The `_transfer_slab!` docstring is correct as written

The original doc rated this MEDIUM and asked for the sentence to be deleted. It is accurate.
`array.jl:606-616`, the `Array ← CuArray` path that `_transfer_slab!` uses for host staging:

```julia
function Base.unsafe_copyto!(dest::Array{T}, doffs, src::DenseCuArray{T}, soffs, n) where T
  context!(context(src)) do
    synchronize(src)      # ← the producing stream, in the producing context
```

`synchronize(x::CuArray) = synchronize(x.data[])` (`array.jl:493`) → `synchronize(managed.stream)`.
That is literally "the D→H copy synchronizes against the stream that produced `src`". Rewriting
it would replace a true statement with a false one.

### Issue #21 was right

#21 states: "Explicit `CUDA.synchronize()` does **not** help — it is not a stream-ordering race."
That is the correct diagnosis. The original doc asked for it to be corrected when closing; do
not. (#21's *other* claim, about `_allgather_x!`, is genuinely stale — see below.)

### Cost of implementing it anyway

The four insertions would not be incorrect, just redundant — each is a full host-side barrier
across all devices, on the Krylov hot path, bought for nothing.

### The one place this reasoning would stop holding

`Managed` mutates `stream`/`dirty` without a lock. `@async` tasks are sticky to one thread, so
they interleave cooperatively and never race on it. **If MDLA ever moves to `Threads.@spawn`,
that becomes a real data race** and the implicit-synchronization guarantee no longer applies.
Worth a comment near the `@sync` blocks if anyone contemplates that change.

---

## Confirmed findings

### Finding A (LOW) — MFO README cites a function that no longer exists

MFO `test/multigpu/README.md` says the fix belongs upstream in MDLA, "bounce the two
cross-device copies through the host, as `_allgather_x!` already does". **`_allgather_x!` was
deleted from MDLA** in commit `a459320` ("feat(ghost): replace allgather with P2P ghost/halo
exchange for SpMV"). MDLA's tracked tree has zero hits for it; the MFO README is now the only
place in either repo that mentions it. Anyone following it will hunt for an implementation to
copy and find nothing.

**The sentence wraps lines 69–70** — line 69 ends at "already", line 70 begins "does)". An edit
targeting line 69 alone will leave a fragment.

### Finding B (LOW) — issue #21's body repeats the stale `_allgather_x!` claim

#21 says "`_allgather_x!` in `src/mul.jl` was already rewritten (71e7f9a) to bounce through the
host". That function no longer exists, so #21's suggested fix isn't actionable as written.
Correct only this when closing — not the synchronization sentence.

### Finding C (HIGH) — test device-count caps hide scale- and topology-dependent bugs

MDLA defaults to every visible GPU in production — `ndevices = length(CUDA.devices())` at
`src/vector.jl:57` and `src/matrix.jl:22`, so 9 on `sasquatch`. All 19 test sites cap far below
(verified exhaustively):
- `min(NGPUS, 4)` — `test_ghost_exchange.jl:3,45,99,156,258,281,358,387`, `test_matrix.jl:2`,
  `test_krylov.jl:2,46,72`, `test_poisson.jl:46`, `test_vector.jl:3`, `test_broadcast.jl:3`
- `min(NGPUS, 2)` — `test_ghost_exchange.jl:202,320,432,488` (`:320` is a
  `compute_partition_ranges(20, min(NGPUS, 2))` call, not an `ndev =` binding like the other three)

The topology gap is the real one and is independent of the refuted race: GPUs 0–3 are all on
NUMA node 0, so a 4-device run never crosses a socket. The `SYS`-class pairs (any of 0–3 against
4–8) have **never** been exercised by any test — and those are exactly the pairs the IOMMU fault
hit hardest.

### Finding D (LOW) — `reduce!` does a redundant double gather

`src/ghost.jl:564-567` gathers `x.partitions[d][ghost.send_indices_gpu[d][k]]` twice — once on
the LHS, once on the RHS — and indexing a `CuArray` with an index array allocates a device
temporary each time. Two allocations per neighbor iteration where the whole read-modify-write
could be one fused kernel. Correctness unaffected.

**Priority note:** `reduce!` has **zero call sites in `src/`**. It is exported at
`MultiDeviceLinearAlgebra.jl:20` and never called internally — neither `mul!` method uses it. Any
`reduce!` work is lower priority than its "HIGH-adjacent" placement in the original doc implied.

### Finding E (LOW) — `scatter!` returns `nothing`

`src/ghost.jl:466` is `return @sync for d in 1:ndevices`, so the function returns the `for`
loop's value (`nothing`) rather than `x`. `reduce!` returns `x` (`:572`). Harmless today —
nothing consumes the return value — but inconsistent and a trap for a future caller.

---

## What's left / next steps

**1. Finish PR #22** (branch `fix/ghost-p2p-fallback`, already open). The finding-3 docstring
rewrite is dropped, so **no code change remains** — `53d8d07` is complete as-is. What's left is
deciding how to land the four unpushed commits listed above (push onto PR #22, or split the
three doc/hygiene ones onto a separate branch off `main`), then merge.

Then close issue #21 with a comment covering: root cause was the host IOMMU in
translating mode, fixed via `intel_iommu=off` + reboot, validated 36/36 pairs; and
`_allgather_x!` no longer exists (replaced by the ghost exchange in `a459320`), so the issue's
suggested fix isn't actionable as written. **Do not claim #21's synchronization conclusion was
wrong.**

Note: MFO has an uncommitted `CLAUDE.md` edit on branch `feat/mdla-ext`, captured by no PR —
decide what to do with it.

**2. Docs.** `docs(readme): document host P2P/IOMMU requirement` — there is **no** "Host
requirements" section today; the natural anchor is the existing `## Requirements` at
`README.md:10` (its body at 12–14 covers Julia/CUDA/GPU count only). Existing P2P mentions are at
`:121`, `:145`, `:168`. Content: PCIe P2P must work; VT-d must be off or in passthrough
(`intel_iommu=off` or `iommu=pt`); `CUDA.can_access_peer` is **not** trustworthy on a
misconfigured host — it returns `true` while transfers silently corrupt; MDLA probes and warns at
`GhostExchange` construction.

**3. MFO** (`~/dev/MatrixFreeOperators`, currently on branch `feat/mdla-ext`).
`docs(test): correct stale _allgather_x! reference in multigpu README` — rewrite
`test/multigpu/README.md:49-70`. **Keep** the `copyto!` reproducer snippet; it's a correct and
useful triage check. Replace the stale half: `_allgather_x!` does not exist; MDLA now probes each
device pair at `GhostExchange` construction and warns + host-stages on failure; the real fix is
host-side.

**4. Repo hygiene.** `gpu-comms-validation/` is **875 MB untracked *and unignored*** inside this
iCloud-synced repo (417M `cuda-samples`, 240M `nccl`, 218M `nccl-tests`, plus nested `.git` dirs
in `cuda-samples/` and `nccl-tests/`). One `git add -A` away from being committed. Move
`gpu-comms-validation/results/` (176 KB, the 4 reports + raw logs) to `docs/gpu-p2p-validation/`;
add `gpu-comms-validation/` and `.DS_Store` to `.gitignore`. The vendored tool checkouts stay
built on `sasquatch` where they're actually run.
`chore(repo): track GPU P2P validation reports, ignore vendored CUDA tooling`.

**5. File two issues.**

*MDLA — `test(ghost): raise device-count caps to NGPUS and cover cross-NUMA pairs`* (HIGH) —
Finding C above. Raise caps to `NGPUS` where tests are device-count agnostic; add an explicit
cross-socket case (e.g. devices 0 and 4).

*MDLA — `perf(ghost): fuse the read-modify-write in reduce!`* (LOW) — Finding D above. Mention
that `reduce!` currently has no internal callers.

*MFO — comment on existing issue #30*, don't open a new one: `test/mdla_gpu.jl:153` caps at
`min(NGPUS_MDLA, 2)` — same coverage gap downstream, and #30 already covers instantiating
`test/multigpu` in CI for ≥3 GPUs.

## Gotchas / constraints

- **Do not revert PR #22.** Kyle decided to keep the probe + host-staging fallback as
  defense-in-depth. It is not dead code to clean up.
- **Do not resurrect the stream-race fix.** If you re-derive the argument from
  `Base.@sync` semantics, read the refutation above first — the answer is in CUDA.jl's
  `Managed` memory layer, not in MDLA.
- **Do not use JuliaFormatter.** CI enforces Runic: `julia -m Runic --inplace .`
- **The refutation is verifiable on macOS** (it's a source read of CUDA.jl). The *runtime*
  verification below still needs the `sasquatch` 9-GPU host.
- **No secrets involved** in any of this work; nothing was redacted from this doc.
- Per repo convention: no "Test plan" section in PR descriptions.

### Verifying the refutation (no GPU needed)

```bash
grep -n -A4 "accessing memory on another stream" ~/.julia/packages/CUDA/*/src/memory.jl
grep -n -B2 -A4 "synchronize(src)" ~/.julia/packages/CUDA/*/src/array.jl
grep -rn "enable_synchronization" src/ test/     # must stay empty
```

### Verification runbook (on `sasquatch`)

```bash
# 1. Full suite at the real device count (9) — the point of the Finding C issue
cd ~/dev/MultiDeviceLinearAlgebra && julia --project -e 'using Pkg; Pkg.test()'

# 2. End-to-end numerical check on a real problem
julia --project scripts/check_poisson.jl

# 3. Confirm the P2P probe is inert — a healthy host must emit NO "silently corrupt data" warning
POISSON_NX=500 julia --project scripts/bench_poisson.jl 2>&1 | grep -i 'corrupt\|warn'

# 4. Downstream: MFO's MDLA extension, which the broken host was blocking
cd ~/dev/MatrixFreeOperators && julia --project=test test/mdla_gpu.jl
```

Step 3 should print nothing. If the probe warns on a host that just passed a 36/36 `simpleP2P`
sweep, **the probe itself is suspect** and should be investigated before trusting the fallback.

### Open, host-side, not code

The post-fix report flags an inventory discrepancy: internal notes describe `sasquatch` as a
10-GPU server, but `nvidia-smi` enumerates 9 after a clean reboot. Either a card is off the PCIe
bus or the note is wrong. Worth a look when convenient.
