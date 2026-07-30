---
slug: issue-27-partition-sensitivity
created: 2026-07-29-1745
status: open
---

# Handoff: finish issue #27 — run the diagnostic on the GPU host, then land Phase B

## Goal / why this matters

Issue #27 ("CG iteration count is partition-dependent") is **misdiagnosed in its own body**. The
358→451→455 iteration spread it reports is not a multi-device defect and not "26% wasted work" — it
is an artifact of the benchmark's own problem setup, which a single-threaded CPU CG reproduces
almost exactly. Phase A of the fix (a diagnostic script) is written and committed to the working
tree. Phase B (tests, benchmark, docs) is deliberately **gated on running that script on the GPU
host**, because the one thing that could still overturn the analysis — whether the halo exchange is
actually correct — cannot be checked from a machine without GPUs.

The next agent's job: get the diagnostic output from `sasquatch`, read it, then land Phase B against
measured numbers instead of inference.

## Background & current state

### What was established (CPU-side, reproducible without a GPU)

1. **`benchmark/gpu.jl:305` uses an RHS that is an exact eigenvector of its own matrix.**
   `b = 2π²·sin(πih)·sin(πjh)` is an eigenvector of the 5-point Laplacian `poisson_matrix_2d`
   builds. Measured: `‖Ab − λb‖/‖Ab‖ = 3.1e-11` at 1000², with `λ` matching the analytic
   `8sin²(πh/2)/h²` to 10 digits. **Exact CG converges in one iteration.**

2. **The requested tolerance is below the attainable floor.** At 2000², `atol = rtol = 1e-12` asks
   for an absolute residual ≈ 2e-8; the true-residual floor is `ε‖A‖‖x‖ ≈ 7e-6`. CG therefore never
   converges on the true residual — it stops when the *recursively updated* residual drifts under
   the threshold, hundreds of iterations later, on a path set entirely by rounding. The issue's own
   table corroborates this: its reported relative residual (2.11e-10 × ‖b‖ ≈ 4e-6) sits at the
   floor, ~100× above what the solver claimed to have reached.

3. **A plain CPU CG reproduces the counts.** Same matrix, same RHS, same tolerance, single thread,
   no GPU: **274 iterations at 2000²**, **15 at 1000²** — against the single-GPU 358 and 25. So
   "358 on one device" is *already* the noise artifact. 451 on two devices is the same artifact with
   a different rounding pattern. Nothing is being wasted by partitioning.

4. **Host-level reduction order is not the mechanism.** Splitting `dot` into 1/2/4/8 contiguous
   partial sums — exactly what `src/vector_linalg.jl:23` does across devices — changed the CPU
   iteration count by **zero** (274 in every configuration, and 15 at 1000² in every configuration).
   So the GPU spread must originate *inside* the partitions: CUBLAS's reduction tree depends on
   partition length, and cuSPARSE's CSR row-block schedule depends on local row count. Neither is
   reachable from Julia.

   **Consequence:** `@test allequal(iter_counts)` at `test/test_krylov.jl:81` asserts an invariant
   that cannot be made true without reproducible summation inside both vendor libraries. The issue's
   suggestion #2 ("a deterministic fixed-order tree reduction over partitions") would not work —
   measurement shows the top-level reduction contributes nothing.

5. **The stream-race theory stays refuted.** Already settled in
   `handoffs/2026-07-27-1604-gpu-comms-validation-findings.md`: CUDA.jl's `Managed` wrapper
   implicitly synchronizes on cross-stream and cross-device access
   (`~/.julia/packages/CUDA/*/src/memory.jl:549,571` — re-verified in every CUDA.jl copy in the
   local depot), and MDLA never calls `enable_synchronization!`. Do not re-derive this.

### What is genuinely still open

Whether the halo exchange is *correct*. An expert colleague suggested the right instrument on the
issue thread: check a 2×2 system where each GPU holds a single DOF. With small-integer entries the
SpMV arithmetic is exact, so any deviation is communication with zero rounding confound. That idea
is what §1 and §2 of the diagnostic implement.

### Repo state

Branch `main`, clean except for **one untracked file**:

- `scripts/diagnose_partition_sensitivity.jl` — Phase A, ~650 lines, Runic-clean, parses. Uses only
  dependencies already in `benchmark/Project.toml` (no new deps).

Nothing else has been touched. No commit, no branch, no PR yet.

Related open issues: **#27** (this one), **#30** (the scatter!/reduce! Phase 1→2 sync question — §2
of the diagnostic doubles as the regression test it asks for), **#31** (unrelated).

## Key files / locations

| Path | Why it matters |
|---|---|
| `scripts/diagnose_partition_sensitivity.jl` | Phase A. The evidence gate. Untracked. |
| `benchmark/gpu.jl:305` | The eigenvector RHS. `benchmark/gpu.jl:265,273` are the `1e-12` tolerances. |
| `test/test_krylov.jl:64-82` | The `allequal(iter_counts)` testset that asserts a false invariant. |
| `src/vector_linalg.jl:23-44` | `dot` / `norm` — sum of per-device CUBLAS partials. Measurement says this is *not* the culprit. |
| `src/mul.jl:1-18` | SpMV: `scatter!` then per-device `mul!`. cuSPARSE row-block schedule lives under this. |
| `src/ghost.jl:502-546` | `scatter!`, the two-phase exchange §1/§2 probe. |
| `handoffs/2026-07-27-1604-*.md` | Prior investigation. Refutes the stream race; Finding C flags the cross-socket test gap the dense pattern now closes. |
| `~/.claude/plans/address-issue-27-and-expressive-flamingo.md` | The plan file. **Not committed** — this handoff supersedes it. |

## Decisions & conclusions

Settled with Kyle, do not relitigate:

- **Sequencing: diagnostic first, as an evidence gate.** Phase B does not start until the diagnostic
  has run on hardware. Chosen over landing everything at once.
- **The `allequal` invariant gets split, not deleted.** Keep exact equality where it is
  mathematically defensible (the one-DOF-per-device exact system); add a well-posed Poisson case
  asserting counts within a *tolerance band* plus cross-device solution agreement. Neither test
  asserts something false, and neither drops the regression guard entirely.
- **The benchmark gets fixed, not left alone.** Fixed-seed random RHS, `rtol = 1e-8`, plus the
  `ms/iter` column the issue itself identified as the real signal, plus printing the true residual
  next to the estimated floor so an unattainable tolerance is visible on sight. Past numbers become
  non-comparable — that is the point.
- **No `src/` change.** No defect was found. Sorting or Kahan-summing `sum(partial)` in
  `src/vector_linalg.jl` would address a mechanism that measurement shows contributes nothing.

## What's left / next steps

1. **Run the diagnostic on `sasquatch`.** Kyle has to do this, or it has to run on the GPU host:

   ```bash
   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
   DIAG_NX=1000 julia --project=benchmark scripts/diagnose_partition_sensitivity.jl   # ~5 min
   DIAG_NX=2000 julia --project=benchmark scripts/diagnose_partition_sensitivity.jl   # ~15-20 min
   ```

   `DIAG_NRUNS=1` roughly halves §5 if the 2000² run drags. Other knobs: `DIAG_NDEVICES`,
   `DIAG_REPEATS`.

2. **Read the gate.** The script exits **nonzero** if §1 or §2 fail.
   - **§1 or §2 FAIL** → the analysis above is overturned. Communication is broken; stop, treat it
     as a correctness bug, and do not touch tests or benchmarks until it is fixed.
   - **Both PASS** → the halo exchange is exact and deterministic. Everything in §3–§5 is
     floating-point re-association and problem conditioning. Proceed to step 3.

3. **Record the numbers that Phase B needs:**
   - §3's verdict column: `reduction only` vs `SpMV too`. Prediction from point 4 above is
     `SpMV too`. Either way it goes in the README note as the reason invariance is unachievable.
   - §5(b)'s iteration spread on the well-posed problem → **calibrates the tolerance band** for the
     reworked `test_krylov.jl`.
   - §5(c)'s `allequal` verdict for both the tridiagonal and dense systems → decides whether the
     exact-system `allequal` assertion can be kept (expected yes; the dense matrix has only two
     distinct eigenvalues, `{n, 2n}`, so exact CG terminates in two steps regardless of partition).

4. **Land Phase B** on a branch off `main`:
   - `test/test_krylov.jl` — replace `allequal(iter_counts)` per the split decision above.
   - Promote diagnostic §1 and §2 into permanent tests (new `test/` file, or fold into
     `test_matrix.jl`; add to `test/runtests.jl` under the `HAS_CUDA` gate). §2 also closes the
     regression-test ask in **#30**.
   - `benchmark/gpu.jl` — the RHS/tolerance/`ms/iter` change described above.
   - `README.md` + `CLAUDE.md` — a "reproducibility across device counts" note: results are not
     bitwise reproducible across `ndevices`, why (CUBLAS tree shape, cuSPARSE row-block schedule),
     and the consequence for choosing tolerances and reading scaling numbers.
   - Decide whether `scripts/diagnose_partition_sensitivity.jl` stays (recommend yes, alongside
     `scripts/check_poisson.jl`, referenced from the README).

5. **Comment on and close #27** with the diagnostic output. The headline for the thread: the
   iteration counts were measuring rounding noise on a problem whose exact answer takes one
   iteration, and a CPU reference reproduces them. Per Kyle's global convention, end any GitHub
   comment posted under his account with a one-line Claude attribution footer (improvise a fresh
   quip; do not reuse a canned string).

## Gotchas / constraints

- **Format with Runic, never JuliaFormatter.** `julia --project=@runic -m Runic --inplace .` — the
  `@runic` shared env already exists in this depot. CI enforces it.
- **No secrets are involved anywhere in this work.** Nothing was redacted from this doc; there was
  nothing to redact. The GPU host is referred to by hostname only.
- **A green test suite on a CPU box proves nothing here.** Only `test_partition.jl`,
  `test_ghost.jl`, and `test_poisson.jl`'s CPU portion run without CUDA.
- **`benchmark/benchmarks.jl` is a different tier** and must not gain anything that does not also
  exist on a PR's base branch — `benchpkg` runs the same file against both revisions. The diagnostic
  is a `scripts/` artifact and stays out of that suite.
- **Dead end already walked: the CUDA stream race.** `@sync` waits on Julia tasks, not CUDA streams,
  and it is tempting to conclude `scatter!`'s Phase 1→2 boundary is unsynchronized. It is not — see
  point 5 above and the prior handoff. Do not insert `CUDA.synchronize()` calls on the hot path.
- **Dead end already walked: fixing the host-side reduction.** Making `sum(partial)` in
  `src/vector_linalg.jl:32` deterministic or fixed-order changes nothing; the CPU experiment showed
  a zero-iteration effect across 1/2/4/8 partitions.
- **`scripts/check_poisson.jl` and `test/test_poisson.jl` use the same eigenvector RHS** at
  `rtol = 1e-12`. At their small grid sizes (100², 30²) the tolerance is still above the floor, so
  they are fine and are *correctness* checks, not scaling ones — leave them alone. The degeneracy
  only bites at large `NX`.
- **The diagnostic rebuilds the full matrix once per device count** and calls `free_devices()`
  between sections; at `DIAG_NX=2000` that is a few hundred MB per rebuild. If it OOMs anyway,
  narrow `DIAG_NDEVICES` rather than assuming a leak.
