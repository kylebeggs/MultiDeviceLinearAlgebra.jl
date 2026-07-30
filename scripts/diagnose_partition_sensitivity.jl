# Diagnostic for issue #27 — why does the CG iteration count depend on the device count?
#
# Run by hand on GPU hardware; there is no CI tier that can execute it:
#
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   DIAG_NX=1000 julia --project=benchmark scripts/diagnose_partition_sensitivity.jl
#   DIAG_NX=2000 julia --project=benchmark scripts/diagnose_partition_sensitivity.jl
#
# Env knobs:
#   DIAG_NX         Poisson grid edge; the problem is NX² unknowns       (default 1000)
#   DIAG_NDEVICES   comma-separated device counts to sweep               (default 1,2,4,<all>)
#   DIAG_REPEATS    back-to-back SpMV calls in the determinism probe     (default 200)
#   DIAG_NRUNS      timed solve repetitions, median is reported          (default 3)
#
# Sections 1 and 2 are correctness assertions and exit nonzero on failure. Sections 0, 3, 4 and 5
# only measure — they never fail, because their job is to produce numbers, not verdicts.
#
# What each section settles:
#
#   §0  Is host P2P healthy? Everything downstream is meaningless if it is not.
#   §1  Is the halo exchange *exact*? One DOF per GPU with small-integer entries makes the SpMV
#       arithmetic exact, so any deviation is communication — no rounding confound at all. Run
#       against two sparsity patterns: tridiagonal (neighbouring device pairs only) and dense
#       (every ordered pair, which is the only case here that crosses the socket).
#   §2  Is it stable under back-to-back SpMV with no host synchronization in between? That is the
#       access pattern CG produces, and the one a stale-buffer race would exploit.
#   §3  Where does partition dependence actually enter — the `dot`/`norm` reduction, or the SpMV?
#       Issue #27 hypothesises the former. §3 measures both, in ULP.
#   §4  Is the benchmark's own problem degenerate? Prints whether `b` is an eigenvector of `A`, and
#       whether the requested tolerance is even attainable in Float64.
#   §5  Iteration counts across device counts for three setups: the benchmark's, a well-posed one,
#       and the exact one-DOF-per-device system.
using CUDA
using LinearAlgebra
using MultiDeviceLinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics

const MDLA = MultiDeviceLinearAlgebra

CUDA.functional() || error("scripts/diagnose_partition_sensitivity.jl needs a functional CUDA installation")

const NGPUS = Int(length(CUDA.devices()))
const NX = parse(Int, get(ENV, "DIAG_NX", "1000"))
const REPEATS = parse(Int, get(ENV, "DIAG_REPEATS", "200"))
const NRUNS = parse(Int, get(ENV, "DIAG_NRUNS", "3"))
const DEVICE_COUNTS = if haskey(ENV, "DIAG_NDEVICES")
    sort!(unique(filter(<=(NGPUS), parse.(Int, split(ENV["DIAG_NDEVICES"], ",")))))
else
    sort!(unique(filter(<=(NGPUS), [1, 2, 4, NGPUS])))
end

const FAILURES = String[]

record_failure(msg) = (push!(FAILURES, msg); nothing)

header(title) = (println(); println("═"^92); println(title); println("═"^92))
subheader(title) = (println(); println(title); println("─"^92))

fmt(t) = t < 1.0e-3 ? @sprintf("%8.1f µs", 1.0e6 * t) : @sprintf("%8.2f ms", 1.0e3 * t)

"""
    sync_devices(spec)

Block until every device backing `spec` has drained. `CUDA.synchronize()` covers only the current
device, so checking the result of a `@sync`/`@async` multi-device call without this can read memory
the producing device has not finished writing.
"""
function sync_devices(spec::PartitionSpec)
    prev = CUDA.device()
    for d in 1:spec.ndevices
        CUDA.device!(device_id(spec, d))
        CUDA.synchronize()
    end
    CUDA.device!(prev)
    return nothing
end

"""
    free_devices()

Drop unreferenced device memory on every GPU. Each section rebuilds the whole matrix once per
device count and holds the previous build only until the next loop iteration, which at
`DIAG_NX=2000` is a few hundred MB per rebuild. CUDA.jl's pool does collect under pressure, but
doing it between sections keeps an unrelated OOM from being mistaken for a diagnostic result.
"""
function free_devices()
    GC.gc(true)
    prev = CUDA.device()
    for dev in 0:(NGPUS - 1)
        CUDA.device!(dev)
        CUDA.reclaim()
    end
    CUDA.device!(prev)
    return nothing
end

"""
    ulp_distance(a, b) -> Int

Number of representable `Float64`s between `a` and `b`; `0` iff they are bit-identical. Maps each
float to a monotone integer ordering first, so the count is meaningful across zero and across
exponent boundaries. `Int128` intermediates keep the negative half from overflowing.
"""
function ulp_distance(a::Float64, b::Float64)
    (isnan(a) || isnan(b)) && return typemax(Int)
    ord(x) = let u = reinterpret(UInt64, x)
        (u & 0x8000_0000_0000_0000) != 0 ? -Int128(u & 0x7fff_ffff_ffff_ffff) : Int128(u)
    end
    return Int(abs(ord(a) - ord(b)))
end

max_ulp(x::AbstractVector, y::AbstractVector) =
    isempty(x) ? 0 : maximum(ulp_distance(xi, yi) for (xi, yi) in zip(x, y))

count_differing(x::AbstractVector, y::AbstractVector) = count(!=(0), x .- y)

# ── §0. Environment ───────────────────────────────────────────────────────────

function section_environment()
    header("§0  Device inventory and P2P health")
    for (i, dev) in enumerate(CUDA.devices())
        @printf("  [%d] %-32s %6.1f GB\n", i - 1, CUDA.name(dev), CUDA.totalmem(dev) / 2^30)
    end

    # A host with broken P2P degrades every ghost transfer to host staging, which reads as an
    # algorithmic slowdown, and — before the probe existed (#21) — read as numerical garbage.
    println("\n  `_p2p_copy_ok` probe — rows are source device, columns destination:")
    print("        ")
    for j in 0:(NGPUS - 1)
        @printf("%4d", j)
    end
    println()
    degraded = 0
    for i in 0:(NGPUS - 1)
        @printf("    %3d ", i)
        for j in 0:(NGPUS - 1)
            if i == j
                print("   ·")
            elseif MDLA._p2p_copy_ok(i, j)
                print("   ✓")
            else
                print("   ✗")
                degraded += 1
            end
        end
        println()
    end
    if degraded > 0
        # `@printf` needs a literal format string — a `*`-concatenated one fails at macro
        # expansion, before the branch can even be reached.
        @printf("\n  ⚠ %d ordered device pairs fall back to host staging. Every number\n", degraded)
        println("    below is still correct, but timings are not representative of healthy")
        println("    hardware.")
    end
    @printf(
        "\n  grid: %d×%d = %d unknowns    device counts swept: %s\n",
        NX, NX, NX * NX, DEVICE_COUNTS
    )
    return nothing
end

# ── §1. Is the halo exchange exact? ───────────────────────────────────────────

"""
    tridiagonal_spd(n)

`n×n` SPD tridiagonal with entries drawn from `{-1, 4}`. Every value, and every partial sum of
`A * (1:n)`, is an integer exactly representable in `Float64` — so the SpMV result is exact
regardless of accumulation order, on any device count. That is what makes a mismatch attributable
to communication rather than to rounding.
"""
function tridiagonal_spd(n::Int)
    e = fill(-1.0, max(n - 1, 0))
    return spdiagm(-1 => e, 0 => fill(4.0, n), 1 => e)
end

"""
    dense_spd(n)

`n×n` SPD matrix `n·I + ones(n, n)`, exactly representable and exactly summable against `1:n` for
any `n` the sweep can reach. Where [`tridiagonal_spd`](@ref) only makes each device talk to its
immediate neighbours, this one is fully dense: at one row per device *every ordered device pair*
exchanges exactly one value. That is the only configuration in the repository that exercises the
cross-socket (`SYS`-class) pairs — the ones the IOMMU fault behind #21 hit hardest.
"""
dense_spd(n::Int) = sparse(fill(1.0, n, n) + n * I)

"""
    diagnose_exact_mismatch(got, expected, A_cpu, x_cpu)

Decode *how* an exact SpMV went wrong. Each failure mode leaves a distinguishable exact vector, so
the diagnosis is a comparison rather than a guess.
"""
function diagnose_exact_mismatch(got, expected, A_cpu, x_cpu)
    diag_only = Vector(diag(A_cpu)) .* x_cpu
    got == diag_only && return "ghost values arrived as ZERO — off-diagonal contributions are " *
        "entirely missing (broken P2P that the construction probe did not catch, or an empty " *
        "send buffer)"
    all(iszero, got) && return "the whole result is zero — the SpMV never ran, or `local_x` was " *
        "never populated"
    wrong = findall(got .!= expected)
    return "$(length(wrong)) of $(length(got)) entries wrong at indices $(first(wrong, 8))" *
        (length(wrong) > 8 ? " …" : "") *
        " — consistent with a stale buffer or a bad local-column remap, not with rounding " *
        "(this arithmetic has no rounding)"
end

"""
    exact_spmv_probe(label, build, ndev)

One exact SpMV at `ndev` devices with `n = ndev`, so every device owns exactly one row and every
off-diagonal entry is a ghost. Prints a row of the §1 table and records a failure if the bitwise
comparison does not hold.
"""
function exact_spmv_probe(label, build, ndev)
    n = ndev
    A_cpu = build(n)
    x_cpu = collect(1.0:n)
    y_exact = A_cpu * x_cpu

    A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
    spec = A_md.row_spec
    x_md = MultiDeviceVector(x_cpu, spec)
    y_md = MultiDeviceVector(zeros(n), spec)

    mul!(y_md, A_md, x_md)
    sync_devices(spec)
    got = gather(y_md)

    pairs = sum(length, A_md.ghost_exchange.ghost_global_indices)
    ok = got == y_exact
    @printf(
        "  %-13s %-9d %-8d %-9s %-40s %s\n",
        label, ndev, pairs, ok ? "PASS" : "FAIL", string(y_exact), string(got)
    )

    if !ok
        println("      ↳ ", diagnose_exact_mismatch(got, y_exact, A_cpu, x_cpu))
        for d in 1:ndev
            @printf(
                "        device %d (CUDA %d) owns rows %s, ghosts %s\n",
                d, device_id(spec, d), string(spec.ranges[d]),
                string(A_md.ghost_exchange.ghost_global_indices[d])
            )
        end
        record_failure("§1 exact SpMV mismatch, $label pattern, ndevices=$ndev")
    end
    return ok
end

function section_exact_communication()
    header("§1  Is the halo exchange exact?  (one DOF per GPU, integer arithmetic)")
    println(
        "  n = ndevices, so every device owns exactly one row and every off-diagonal entry is a\n" *
            "  ghost — the highest communication-to-compute ratio the package can be given. All\n" *
            "  arithmetic is exact in Float64, so the comparison below is bitwise `==`, not `≈`,\n" *
            "  and any mismatch is communication with no rounding to hide behind.\n" *
            "\n" *
            "  Two sparsity patterns, because they cover different device pairs:\n" *
            "    tridiagonal   each device talks only to its immediate neighbours\n" *
            "    dense         every ordered device pair exchanges a value — the only case here\n" *
            "                  that crosses the socket on a multi-NUMA host"
    )

    subheader(
        @sprintf(
            "  %-13s %-9s %-8s %-9s %-40s %s",
            "pattern", "devices", "ghosts", "verdict", "expected  (bitwise)", "got"
        )
    )

    for ndev in DEVICE_COUNTS
        exact_spmv_probe("tridiagonal", tridiagonal_spd, ndev)
        # A dense n×n at n = 1 is the tridiagonal, and at n < 2 there is nothing to exchange.
        ndev >= 2 && exact_spmv_probe("dense", dense_spd, ndev)

        # The 2×2 case named in the issue thread, spelled out so the output stands on its own.
        if ndev == 2
            println(
                "      ↳ the 2×2 case from the issue thread: A = [4 -1; -1 4], x = [1, 2],\n" *
                    "        each GPU holding one DOF and exchanging exactly one value. Correct is\n" *
                    "        [2, 7]; ghosts arriving as zero would give [4, 8]."
            )
        end
    end
    return nothing
end

# ── §2. Is it stable under back-to-back SpMV? ─────────────────────────────────

function section_determinism()
    header("§2  Is the exchange stable under back-to-back SpMV with no host sync?")
    println(
        "  CG issues `mul!` after `mul!` with nothing draining the devices in between, and every\n" *
            "  call reuses the same send/recv/`local_x` buffers. A stale-buffer race would show up\n" *
            "  here and nowhere else — the existing tests gather after every call, which inserts the\n" *
            "  very synchronization that would hide it."
    )

    # (a) Exact system, alternating inputs. Alternating is what makes staleness visible: a call that
    #     reads the previous iteration's send buffer produces the *other* vector's answer, and the
    #     two answers differ in every entry. Dense rather than tridiagonal, so every ordered device
    #     pair is under load at once.
    subheader(@sprintf("  (a) one DOF per GPU, dense, %d alternating calls, bitwise vs exact", REPEATS))
    @printf("  %-9s %-9s %-12s %s\n", "devices", "verdict", "mismatches", "first bad call")
    for ndev in DEVICE_COUNTS
        ndev < 2 && continue
        n = ndev
        A_cpu = dense_spd(n)
        xa_cpu = collect(1.0:n)
        xb_cpu = collect(Float64, n:-1:1)
        ya, yb = A_cpu * xa_cpu, A_cpu * xb_cpu

        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
        spec = A_md.row_spec
        xa = MultiDeviceVector(xa_cpu, spec)
        xb = MultiDeviceVector(xb_cpu, spec)
        ys = [MultiDeviceVector(zeros(n), spec) for _ in 1:REPEATS]

        for k in 1:REPEATS
            mul!(ys[k], A_md, isodd(k) ? xa : xb)
        end
        sync_devices(spec)

        bad = [k for k in 1:REPEATS if gather(ys[k]) != (isodd(k) ? ya : yb)]
        @printf(
            "  %-9d %-9s %-12s %s\n",
            ndev, isempty(bad) ? "PASS" : "FAIL", "$(length(bad))/$REPEATS",
            isempty(bad) ? "—" : string(first(bad))
        )
        isempty(bad) || record_failure("§2a nondeterministic SpMV at ndevices=$ndev")
    end

    # (b) Real halo, real problem size. The arithmetic is no longer exact, so the assertion is
    #     self-consistency: the same input through the same operator must give the same bits every
    #     time. Any variation is nondeterminism, whatever its source.
    reps_b = min(REPEATS, 25)
    subheader(@sprintf("  (b) %d×%d Poisson halo, %d identical calls, bitwise self-consistency", NX, NX, reps_b))
    @printf("  %-9s %-9s %-12s %s\n", "devices", "verdict", "mismatches", "max ULP vs call 1")
    A_cpu = poisson_matrix_2d(NX, NX)
    Random.seed!(0x00C0FFEE)
    x_cpu = randn(NX * NX)
    for ndev in DEVICE_COUNTS
        ndev < 2 && continue
        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
        spec = A_md.row_spec
        x_md = MultiDeviceVector(x_cpu, spec)
        ys = [MultiDeviceVector(zeros(NX * NX), spec) for _ in 1:reps_b]

        for k in 1:reps_b
            mul!(ys[k], A_md, x_md)
        end
        sync_devices(spec)

        ref = gather(ys[1])
        worst = 0
        bad = Int[]
        for k in 2:reps_b
            yk = gather(ys[k])
            if yk != ref
                push!(bad, k)
                worst = max(worst, max_ulp(yk, ref))
            end
        end
        @printf(
            "  %-9d %-9s %-12s %d\n",
            ndev, isempty(bad) ? "PASS" : "FAIL", "$(length(bad))/$(reps_b - 1)", worst
        )
        isempty(bad) || record_failure("§2b nondeterministic SpMV at ndevices=$ndev, NX=$NX")
        free_devices()
    end
    return nothing
end

# ── §3. Where does partition dependence enter? ────────────────────────────────

function section_partition_sensitivity()
    header("§3  Where does partition dependence enter — the reduction, or the SpMV?")
    println(
        "  Issue #27 attributes the whole effect to `dot`/`norm` reduction order. If that is the\n" *
            "  complete story, SpMV agrees bitwise across device counts (0 ULP) and only the scalar\n" *
            "  reductions move. If SpMV also moves, cuSPARSE's row-block schedule is partition-\n" *
            "  dependent too, and no host-side reduction fix could ever make the solve invariant.\n" *
            "  Everything below is measured against the single-device result on identical input."
    )

    A_cpu = poisson_matrix_2d(NX, NX)
    Random.seed!(0x00C0FFEE)
    x_cpu = randn(NX * NX)

    ref_spec = compute_partition_ranges(NX * NX, 1)
    A_ref = MultiDeviceSparseMatrixCSR(A_cpu, ref_spec)
    x_ref = MultiDeviceVector(x_cpu, ref_spec)
    y_ref_md = MultiDeviceVector(zeros(NX * NX), ref_spec)
    mul!(y_ref_md, A_ref, x_ref)
    sync_devices(ref_spec)
    y_ref = gather(y_ref_md)
    dot_ref = dot(x_ref, x_ref)
    norm_ref = norm(x_ref)

    subheader(
        @sprintf(
            "  %-9s %-10s %-11s %-13s %-13s %-13s %s",
            "devices", "dot ULP", "norm ULP", "SpMV max ULP", "SpMV ‖Δ‖/‖y‖", "SpMV entries≠", "verdict"
        )
    )
    for ndev in DEVICE_COUNTS
        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
        spec = A_md.row_spec
        x_md = MultiDeviceVector(x_cpu, spec)
        y_md = MultiDeviceVector(zeros(NX * NX), spec)
        mul!(y_md, A_md, x_md)
        sync_devices(spec)
        y = gather(y_md)

        d_ulp = ulp_distance(dot(x_md, x_md), dot_ref)
        n_ulp = ulp_distance(norm(x_md), norm_ref)
        y_ulp = max_ulp(y, y_ref)
        y_rel = norm(y - y_ref) / norm(y_ref)
        y_cnt = count_differing(y, y_ref)

        verdict = if ndev == 1
            "reference"
        elseif y_ulp == 0 && d_ulp == 0 && n_ulp == 0
            "bit-identical"
        elseif y_ulp == 0
            "reduction only"
        else
            "SpMV too"
        end
        @printf(
            "  %-9d %-10d %-11d %-13d %-13.3e %-13d %s\n",
            ndev, d_ulp, n_ulp, y_ulp, y_rel, y_cnt, verdict
        )
        free_devices()
    end
    println(
        "\n  A few ULP is ordinary floating-point re-association. A large `‖Δ‖/‖y‖` (say > 1e-12)\n" *
            "  would mean something other than rounding is at work — but §1 and §2 would have caught\n" *
            "  that first, with no rounding in the way."
    )
    return nothing
end

# ── §4. Is the benchmark's own problem degenerate? ────────────────────────────

function section_problem_conditioning()
    header("§4  Is the benchmark's problem well posed for a 1e-12 tolerance?")

    A_cpu = poisson_matrix_2d(NX, NX)
    h = 1.0 / (NX + 1)
    b_eig = [2π^2 * sin(π * i * h) * sin(π * j * h) for j in 1:NX for i in 1:NX]

    Ab = A_cpu * b_eig
    λ_est = dot(Ab, b_eig) / dot(b_eig, b_eig)
    λ_exact = 8 * sin(π * h / 2)^2 / h^2
    eig_defect = norm(Ab - λ_est * b_eig) / norm(Ab)

    normb = norm(b_eig)
    x_exact = b_eig / λ_est
    normA = maximum(sum(abs, A_cpu; dims = 2))
    floor_abs = eps(Float64) * normA * norm(x_exact)
    requested = 1.0e-12 + 1.0e-12 * normb

    println("  `benchmark/gpu.jl` uses b = 2π²·sin(πx)sin(πy) with the 5-point Laplacian.")
    @printf("    λ from ⟨Ab,b⟩/⟨b,b⟩            %.12f\n", λ_est)
    @printf("    λ exact, 8sin²(πh/2)/h²        %.12f\n", λ_exact)
    @printf("    ‖Ab − λb‖ / ‖Ab‖               %.3e   ← b is an eigenvector iff this is ~0\n", eig_defect)
    println()
    @printf("    ‖b‖                            %.4e\n", normb)
    @printf("    requested ε (atol+rtol·‖b‖)    %.4e   at atol = rtol = 1e-12\n", requested)
    # Order-of-magnitude estimate, not a bound: the standard ε‖A‖‖x‖ backward-error rule of thumb,
    # with ‖A‖ taken as the max row sum. It only has to be right to a factor of a few to settle
    # whether the requested tolerance is on the achievable side of the line.
    @printf("    attainable floor ≈ ε_mach‖A‖‖x‖ %.4e\n", floor_abs)
    println()
    if eig_defect < 1.0e-8
        println(
            "  ⇒ b IS an eigenvector of A, so exact CG converges in ONE iteration. Every iteration\n" *
                "    beyond the first is chasing rounding noise, and its count is a property of the\n" *
                "    rounding pattern — which the device count changes."
        )
    else
        println("  ⇒ b is not an eigenvector (defect $(eig_defect)); the one-iteration argument does not apply.")
    end
    if requested < floor_abs
        @printf("  ⇒ the requested tolerance is %.0f× BELOW the attainable floor. CG\n", floor_abs / requested)
        println("    cannot converge on the true residual and stops only when the recursively")
        println("    updated residual drifts under the threshold — an artifact, not a solve.")
    else
        @printf("  ⇒ the requested tolerance is %.1f× above the floor and is attainable.\n", requested / floor_abs)
    end
    return nothing
end

# ── §5. Iteration counts across device counts ─────────────────────────────────

"""
    solve_row(A_cpu, b_cpu, ndev; atol, rtol)

One timed `mdla_solve` at `ndev` devices, reporting the iteration count alongside both residuals:
the one the solver stopped on (recursively updated) and the one that is actually true. A gap
between them is the signature of a tolerance below the attainable floor.
"""
function solve_row(A_cpu, b_cpu, ndev; atol, rtol)
    A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
    spec = A_md.row_spec
    b_md = MultiDeviceVector(b_cpu, spec)

    mdla_solve(A_md, b_md; atol, rtol)   # warm up and compile
    sync_devices(spec)

    times = Float64[]
    local x_md, stats
    for _ in 1:NRUNS
        sync_devices(spec)
        t = @elapsed begin
            x_md, stats = mdla_solve(A_md, b_md; atol, rtol)
            sync_devices(spec)
        end
        push!(times, t)
    end

    y_md = similar(b_md)
    mul!(y_md, A_md, x_md)
    sync_devices(spec)
    true_res = norm(gather(y_md) - b_cpu) / norm(b_cpu)

    return (
        iters = stats.niter,
        solved = stats.solved,
        wall = median(times),
        true_res = true_res,
    )
end

function iteration_table(title, A_cpu, b_cpu; atol, rtol)
    subheader("  $title   (atol = $atol, rtol = $rtol)")
    @printf(
        "  %-9s %-8s %-9s %-12s %-12s %s\n",
        "devices", "iters", "solved", "wall", "ms/iter", "true rel. residual"
    )
    counts = Int[]
    for ndev in DEVICE_COUNTS
        r = solve_row(A_cpu, b_cpu, ndev; atol, rtol)
        push!(counts, r.iters)
        @printf(
            "  %-9d %-8d %-9s %-12s %-12.4f %.3e\n",
            ndev, r.iters, r.solved, fmt(r.wall), 1.0e3 * r.wall / max(r.iters, 1), r.true_res
        )
        free_devices()
    end
    lo, hi = extrema(counts)
    @printf(
        "  spread: %d–%d iterations, %+.1f%% — %s\n",
        lo, hi, 100 * (hi - lo) / lo,
        lo == hi ? "partition-invariant on this problem" :
            "NOT partition-invariant; a test band must cover at least this"
    )
    return counts
end

function section_iteration_counts()
    header("§5  Iteration count vs device count, three setups")

    A_cpu = poisson_matrix_2d(NX, NX)
    h = 1.0 / (NX + 1)
    b_eig = [2π^2 * sin(π * i * h) * sin(π * j * h) for j in 1:NX for i in 1:NX]
    Random.seed!(0x00C0FFEE)
    b_rand = randn(NX * NX)

    iteration_table(
        "(a) benchmark's setup — eigenvector RHS, tolerance below the floor (reproduces #27)",
        A_cpu, b_eig; atol = 1.0e-12, rtol = 1.0e-12
    )
    iteration_table(
        "(b) well-posed — fixed-seed random RHS, tolerance above the floor",
        A_cpu, b_rand; atol = 0.0, rtol = 1.0e-8
    )

    # (c) is the one place an `allequal` assertion is defensible, and this is the run that decides
    # whether Phase B can keep it. `n = ndevices` is small enough that CG stops on its finite-
    # termination property rather than on a noise-driven residual — decisively so for the dense
    # matrix, whose spectrum is just {n, 2n}, two distinct eigenvalues, so exact CG terminates in
    # two steps no matter how the work is split.
    n_exact = maximum(DEVICE_COUNTS)
    b_small = collect(1.0:n_exact)
    for (label, build) in (("tridiagonal", tridiagonal_spd), ("dense", dense_spd))
        A_small = build(n_exact)
        subheader(
            "  (c) exact system, $label, n = $(n_exact) " *
                "(one DOF per GPU at the widest sweep point)"
        )
        @printf("  %-9s %-8s %-9s %s\n", "devices", "iters", "solved", "true rel. residual")
        counts_c = Int[]
        for ndev in DEVICE_COUNTS
            A_md = MultiDeviceSparseMatrixCSR(A_small; ndevices = ndev)
            spec = A_md.row_spec
            b_md = MultiDeviceVector(b_small, spec)
            x_md, stats = mdla_solve(A_md, b_md; atol = 1.0e-12, rtol = 1.0e-12)
            sync_devices(spec)
            y_md = similar(b_md)
            mul!(y_md, A_md, x_md)
            sync_devices(spec)
            push!(counts_c, stats.niter)
            @printf(
                "  %-9d %-8d %-9s %.3e\n",
                ndev, stats.niter, stats.solved, norm(gather(y_md) - b_small) / norm(b_small)
            )
        end
        @printf(
            "  %s — `allequal(iter_counts)` %s be asserted on the %s system.\n",
            allequal(counts_c) ? "all equal" : "NOT all equal: $counts_c",
            allequal(counts_c) ? "can" : "cannot", label
        )
    end
    return nothing
end

# ── Run ───────────────────────────────────────────────────────────────────────

for section in (
        section_environment,
        section_exact_communication,
        section_determinism,
        section_partition_sensitivity,
        section_problem_conditioning,
        section_iteration_counts,
    )
    section()
    free_devices()
end

header("Summary")
if isempty(FAILURES)
    println(
        "  §1 and §2 passed: the halo exchange is exact and deterministic on this host.\n" *
            "  Whatever §3–§5 show is therefore floating-point re-association and problem\n" *
            "  conditioning, not broken communication."
    )
else
    println("  $(length(FAILURES)) correctness failure(s) — communication is NOT sound on this host:")
    for f in FAILURES
        println("    • ", f)
    end
    println(
        "\n  This overturns the rounding explanation for issue #27. Fix communication before\n" *
            "  reading anything into §3–§5."
    )
    exit(1)
end
