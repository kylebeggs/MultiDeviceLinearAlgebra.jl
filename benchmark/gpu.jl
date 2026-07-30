# Multi-GPU benchmarks. Standalone — NOT part of the AirspeedVelocity `SUITE`, because CI runs
# on GitHub-hosted runners that have no GPU. Run this by hand on GPU hardware:
#
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   POISSON_NX=500 julia --project=benchmark benchmark/gpu.jl
#
# Env knobs:
#   POISSON_NX     grid edge; the problem is NX² unknowns      (default 500)
#   BENCH_NRUNS    timed solve repetitions, median is reported (default 5)
#   BENCH_NDEVICES comma-separated device counts to sweep      (default 1,2,4,<all>)
#
# On device counts: at the 500² default (250k unknowns) the problem is small enough that halo
# communication dominates and multi-GPU scaling is flat to negative. Push POISSON_NX to
# 1500–2000 before drawing conclusions about strong scaling.
#
# Every timed section asserts parity against a reference first, so a run on unfamiliar hardware
# doubles as a smoke test for the paths CI never executes.
#
# NOT COMPARABLE TO NUMBERS RECORDED BEFORE 2026-07-30. The solve benchmark used to take
# b = 2π²·sin(πx)sin(πy), which is an exact eigenvector of the 5-point Laplacian this file
# assembles — exact CG converges on it in one iteration, so every iteration past the first was
# chasing rounding noise. It also asked for atol = rtol = 1e-12, roughly 90× below the attainable
# residual floor at 1000² and further below it as the grid grows, so CG could never converge on
# the true residual and stopped only when the recursively updated one drifted under the
# threshold. Iteration counts from that setup measured the rounding pattern, not the solver, and
# moved with the device count for the same reason (issue #27). The RHS is now a fixed-seed random
# vector at a tolerance above the floor, and the floor is printed next to the residual so an
# unattainable request is visible on sight.
using BenchmarkTools
using CUDA
using LinearAlgebra
using MultiDeviceLinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics

const MDLA = MultiDeviceLinearAlgebra

CUDA.functional() || error("benchmark/gpu.jl needs a functional CUDA installation")

"""
    sync_devices(spec)

Block until every device backing `spec` has drained. `CUDA.synchronize()` covers only the
current device, so timing a `@sync`/`@async` multi-device call without this measures Julia task
scheduling rather than GPU work.
"""
function sync_devices(spec::MDLA.PartitionSpec)
    prev = CUDA.device()
    for d in 1:spec.ndevices
        CUDA.device!(device_id(spec, d))
        CUDA.synchronize()
    end
    CUDA.device!(prev)
    return nothing
end

fmt(t) = t < 1.0e-3 ? @sprintf("%8.1f µs", 1.0e6 * t) : @sprintf("%8.2f ms", 1.0e3 * t)

function fmtbytes(b)
    b == 0 && return "      0 B"
    b < 1024^2 && return @sprintf("%6.1f KB", b / 1024)
    return @sprintf("%6.1f MB", b / 1024^2)
end

const NGPUS = Int(length(CUDA.devices()))
const NX = parse(Int, get(ENV, "POISSON_NX", "500"))
const NRUNS = parse(Int, get(ENV, "BENCH_NRUNS", "5"))
# Above the attainable residual floor at every grid size this file is meant to be run at. See
# the header for what asking for less than the floor did to the iteration counts.
const SOLVE_ATOL = 0.0
const SOLVE_RTOL = 1.0e-8
const DEVICE_COUNTS = if haskey(ENV, "BENCH_NDEVICES")
    sort!(unique(filter(<=(NGPUS), parse.(Int, split(ENV["BENCH_NDEVICES"], ",")))))
else
    sort!(unique(filter(<=(NGPUS), [1, 2, 4, NGPUS])))
end

header(title) = (println("═"^78); println(title); println("═"^78))

# ── 1. Device inventory ───────────────────────────────────────────────────────

function bench_inventory()
    header("Device inventory")
    for (i, dev) in enumerate(CUDA.devices())
        @printf("  [%d] %-32s %6.1f GB\n", i - 1, CUDA.name(dev), CUDA.totalmem(dev) / 2^30)
    end

    # The same cached probe `GhostExchange` construction runs. On a host with broken P2P
    # (IOMMU/ACS misconfiguration) every ghost transfer silently degrades to host staging,
    # which reads as an algorithmic slowdown if you do not know to check here first.
    println("\nP2P copy probe (`_p2p_copy_ok`) — rows are source, columns destination:")
    print("      ")
    for j in 0:(NGPUS - 1)
        @printf("%4d", j)
    end
    println()
    for i in 0:(NGPUS - 1)
        @printf("  %3d ", i)
        for j in 0:(NGPUS - 1)
            print(i == j ? "   ·" : (MDLA._p2p_copy_ok(i, j) ? "   ✓" : "   ✗"))
        end
        println()
    end
    @printf(
        "\ngrid: %d×%d = %d unknowns    device counts swept: %s\n\n",
        NX, NX, NX * NX, DEVICE_COUNTS
    )
    return nothing
end

# ── 2. Fused indexed kernels vs the broadcast they replaced (issue #24) ───────

# The pre-#24 expressions, kept verbatim as reference implementations so the comparison is
# self-contained and does not require checking out an older revision. `getindex(::CuVector,
# ::CuVector)` is evaluated eagerly rather than fused into the broadcast, so each occurrence
# materializes a device temporary: one for the gather, two for the read-modify-write.
_gather_broadcast!(buf, x, idx) = (buf .= x[idx])
_scatter_apply_broadcast!(x, idx, buf, op) = (x[idx] .= op.(x[idx], buf))

function bench_indexed_kernels()
    header("Indexed gather / scatter-apply — fused kernel vs broadcast (single device)")
    @printf(
        "%-10s %-14s %11s %11s %9s   %10s %10s\n",
        "n indices", "op", "fused", "broadcast", "speedup", "fused", "broadcast"
    )
    println("─"^78)

    CUDA.device!(0)
    Random.seed!(0xBEEF)
    for n in (10^3, 10^4, 10^5, 10^6)
        src_len = 4n
        x = CUDA.rand(Float64, src_len)
        # Unique, sorted indices — the precondition `_scatter_apply!` relies on, and what
        # `_compute_ghost_map` actually produces.
        idx = CuVector(sort!(randperm(src_len)[1:n]))
        buf = CUDA.rand(Float64, n)

        b_fused = similar(buf)
        b_bcast = similar(buf)
        MDLA._gather!(b_fused, x, idx)
        _gather_broadcast!(b_bcast, x, idx)
        Array(b_fused) == Array(b_bcast) || error("gather parity failure at n=$n")

        x_fused = copy(x)
        x_bcast = copy(x)
        MDLA._scatter_apply!(x_fused, idx, buf, +)
        _scatter_apply_broadcast!(x_bcast, idx, buf, +)
        Array(x_fused) == Array(x_bcast) || error("scatter-apply parity failure at n=$n")

        t_gf = @belapsed CUDA.@sync(MDLA._gather!($b_fused, $x, $idx)) seconds = 2
        t_gb = @belapsed CUDA.@sync(_gather_broadcast!($b_bcast, $x, $idx)) seconds = 2
        a_gf = CUDA.@allocated MDLA._gather!(b_fused, x, idx)
        a_gb = CUDA.@allocated _gather_broadcast!(b_bcast, x, idx)

        t_sf = @belapsed CUDA.@sync(MDLA._scatter_apply!($x_fused, $idx, $buf, +)) seconds = 2
        t_sb = @belapsed CUDA.@sync(
            _scatter_apply_broadcast!($x_bcast, $idx, $buf, +)
        ) seconds = 2
        a_sf = CUDA.@allocated MDLA._scatter_apply!(x_fused, idx, buf, +)
        a_sb = CUDA.@allocated _scatter_apply_broadcast!(x_bcast, idx, buf, +)

        @printf(
            "%-10d %-14s %11s %11s %8.2f×   %10s %10s\n",
            n, "gather", fmt(t_gf), fmt(t_gb), t_gb / t_gf, fmtbytes(a_gf), fmtbytes(a_gb)
        )
        @printf(
            "%-10s %-14s %11s %11s %8.2f×   %10s %10s\n",
            "", "scatter-apply", fmt(t_sf), fmt(t_sb), t_sb / t_sf,
            fmtbytes(a_sf), fmtbytes(a_sb)
        )
    end
    println()
    return nothing
end

# ── 3. Ghost exchange: scatter! / reduce! ─────────────────────────────────────

function bench_ghost_exchange(A_cpu, b_cpu)
    header("Ghost exchange on the Poisson halo — per-call time and device allocation")
    @printf(
        "%-9s %9s  %11s %11s   %10s %10s\n",
        "devices", "ghosts", "scatter!", "reduce!", "scatter", "reduce"
    )
    println("─"^78)

    for ndev in DEVICE_COUNTS
        # A single-device exchange has no neighbors, hence no packing and no transfers.
        ndev == 1 && continue

        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
        spec = A_md.row_spec
        ghost = A_md.ghost_exchange
        x = attach_ghost(MultiDeviceVector(b_cpu, spec), ghost)
        n_ghost = sum(length, ghost.ghost_global_indices)

        scatter!(x)
        sync_devices(spec)
        for d in 1:ndev
            CUDA.device!(device_id(spec, d))
            n_owned = length(spec.ranges[d])
            Array(ghost.local_x[d])[1:n_owned] == Array(x.partitions[d]) ||
                error("scatter! owned-prefix parity failure at ndev=$ndev, device $d")
        end

        t_s = @belapsed begin
            scatter!($x)
            sync_devices($spec)
        end seconds = 2
        t_r = @belapsed begin
            reduce!($x, +)
            sync_devices($spec)
        end seconds = 2
        a_s = CUDA.@allocated scatter!(x)
        a_r = CUDA.@allocated reduce!(x, +)

        @printf(
            "%-9d %9d  %11s %11s   %10s %10s\n",
            ndev, n_ghost, fmt(t_s), fmt(t_r), fmtbytes(a_s), fmtbytes(a_r)
        )
    end
    println()
    return nothing
end

# ── 4. SpMV ───────────────────────────────────────────────────────────────────

function bench_spmv(A_cpu, b_cpu, y_ref)
    header("SpMV `mul!(y, A, x)` — device scaling (speedup vs the first row)")
    @printf("%-9s %11s %9s   %10s   %s\n", "devices", "time", "speedup", "alloc", "rel. error")
    println("─"^78)

    t_base = 0.0
    for ndev in DEVICE_COUNTS
        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
        spec = A_md.row_spec
        x_md = MultiDeviceVector(b_cpu, spec)
        y_md = MultiDeviceVector{Float64}(undef, spec)

        mul!(y_md, A_md, x_md)
        sync_devices(spec)
        rel = norm(gather(y_md) - y_ref) / norm(y_ref)
        rel < 1.0e-10 || error("SpMV parity failure at ndev=$ndev: rel. error $rel")

        t = @belapsed begin
            mul!($y_md, $A_md, $x_md)
            sync_devices($spec)
        end seconds = 3
        a = CUDA.@allocated mul!(y_md, A_md, x_md)

        ndev == first(DEVICE_COUNTS) && (t_base = t)
        @printf("%-9d %11s %8.2f×   %10s   %9.2e\n", ndev, fmt(t), t_base / t, fmtbytes(a), rel)
    end
    println()
    return nothing
end

# ── 5. Krylov solve ───────────────────────────────────────────────────────────

function bench_solve(A_cpu, b_cpu)
    header("CG solve `mdla_solve(A, b)` — device scaling ($NRUNS runs, median)")
    # `ms/iter` is the honest per-device-count signal. Wall-clock conflates solver speed with
    # iteration count, and the iteration count is not invariant across partitions — cuSPARSE's
    # CSR row-block schedule depends on the local row count, so the rounding path differs. On an
    # ill-posed problem that difference moved the count by 20% and buried a genuine per-iteration
    # speedup (issue #27).
    @printf(
        "%-9s %11s %9s %7s %10s %11s   %s\n",
        "devices", "time", "speedup", "iters", "ms/iter", "VRAM/dev", "rel. residual"
    )
    println("─"^78)

    n = length(b_cpu)
    normb = norm(b_cpu)
    # Max row sum as a stand-in for ‖A‖; only has to be right to a factor of a few to settle
    # whether the requested tolerance is on the achievable side of the line.
    normA = maximum(sum(abs, A_cpu; dims = 2))
    t_base = 0.0
    floor_rel = 0.0
    for ndev in DEVICE_COUNTS
        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
        spec = A_md.row_spec
        b_md = MultiDeviceVector(b_cpu, spec)

        # CSR values (8 B) + column indices (4 B) + row pointers (4 B/row), plus CG's working
        # set of roughly four length-n vectors.
        vram_mb = ((nnz(A_cpu) ÷ ndev) * 12 + (n ÷ ndev) * (4 + 8 * 4)) / 2^20

        mdla_solve(A_md, b_md; atol = SOLVE_ATOL, rtol = SOLVE_RTOL)    # warm up and compile
        sync_devices(spec)

        times = Float64[]
        local x_md, stats
        for _ in 1:NRUNS
            sync_devices(spec)
            t = @elapsed begin
                x_md, stats = mdla_solve(A_md, b_md; atol = SOLVE_ATOL, rtol = SOLVE_RTOL)
                sync_devices(spec)
            end
            push!(times, t)
        end
        t_med = median(times)

        stats.solved || error("CG did not converge at ndev=$ndev")
        y_md = similar(b_md)
        mul!(y_md, A_md, x_md)
        sync_devices(spec)
        rel_res = norm(gather(y_md) - b_cpu) / normb
        floor_rel == 0.0 && (floor_rel = eps(Float64) * normA * norm(gather(x_md)) / normb)

        ndev == first(DEVICE_COUNTS) && (t_base = t_med)
        @printf(
            "%-9d %11s %8.2f× %7d %10.4f %8.1f MB   %9.2e\n",
            ndev, fmt(t_med), t_base / t_med, stats.niter,
            1.0e3 * t_med / max(stats.niter, 1), vram_mb, rel_res
        )
    end
    # A requested tolerance below this floor cannot be met on the true residual; CG would stop
    # only once the recursively updated residual drifted under it, hundreds of iterations later
    # on a path set entirely by rounding.
    @printf(
        "\nrequested rtol %.1e   ·   attainable floor ≈ ε‖A‖‖x‖/‖b‖ = %.2e   ·   %s\n",
        SOLVE_RTOL, floor_rel,
        SOLVE_RTOL > floor_rel ? "attainable" : "BELOW THE FLOOR — counts are rounding artifacts"
    )
    println()
    return nothing
end

# ── Run ───────────────────────────────────────────────────────────────────────

bench_inventory()
bench_indexed_kernels()

print("Assembling $(NX)×$(NX) Poisson matrix... ")
t_asm = @elapsed A_cpu = poisson_matrix_2d(NX, NX)
@printf("%.3f s  (nnz = %d)\n\n", t_asm, nnz(A_cpu))

# Fixed seed so the problem is identical run to run and across device counts. Deliberately not
# the analytic sin·sin forcing: that vector is an eigenvector of this matrix (see the header).
Random.seed!(0x00C0FFEE)
b_cpu = randn(NX * NX)
y_ref = A_cpu * b_cpu

bench_ghost_exchange(A_cpu, b_cpu)
bench_spmv(A_cpu, b_cpu, y_ref)
bench_solve(A_cpu, b_cpu)
