using MultiDeviceLinearAlgebra
using CUDA
using LinearAlgebra
using Printf
using Statistics

function sync_all_devices(n)
    for d in 0:(n - 1)
        CUDA.device!(d)
        CUDA.synchronize()
    end
end

# ── GPU inventory ─────────────────────────────────────────────────────────────

ngpus = length(CUDA.devices())
println("GPUs: $ngpus")
for d in CUDA.devices()
    vram_gb = CUDA.total_memory() / 2^30
    @printf("  %s — %.1f GB VRAM\n", CUDA.name(d), vram_gb)
end
println()

# ── Problem setup ─────────────────────────────────────────────────────────────

nx = ny = parse(Int, get(ENV, "POISSON_NX", "500"))
N = nx * ny
hx = 1.0 / (nx + 1)
hy = 1.0 / (ny + 1)
nruns = parse(Int, get(ENV, "BENCH_NRUNS", "5"))

@printf("2D Poisson: %d×%d grid → %d unknowns\n", nx, ny, N)
@printf("Benchmark runs: %d\n\n", nruns)

print("Assembling matrix... ")
t_asm = @elapsed A_cpu = poisson_matrix_2d(nx, ny)
@printf("%.3f s  (nnz = %d)\n", t_asm, nnz(A_cpu))

b_cpu = zeros(N)
u_exact = zeros(N)
for j in 1:ny, i in 1:nx
    x = i * hx
    y = j * hy
    idx = (j - 1) * nx + i
    b_cpu[idx] = 2π^2 * sin(π * x) * sin(π * y)
    u_exact[idx] = sin(π * x) * sin(π * y)
end
println()

# ── Multi-device sweep ────────────────────────────────────────────────────────

# (ndev, time, iters, vram_per_dev_mb)
results = Tuple{Int,Float64,Int,Float64}[]

for ndev in 1:ngpus
    @printf("── %d device(s) ", ndev)
    println("─" ^ 48)

    # Upload
    print("  Upload... ")
    t_up = @elapsed begin
        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices=ndev)
        b_md = MultiDeviceVector(b_cpu; ndevices=ndev)
    end
    @printf("%.3f s\n", t_up)

    # VRAM estimate per device (matrix nnz + vector elements, Float64 values + Int32 indices)
    nnz_total = nnz(A_cpu)
    nnz_per_dev = nnz_total ÷ ndev
    n_per_dev = N ÷ ndev
    # CSR: nzval(8B) + colval(4B) + rowptr(4B per row) + vectors
    vram_mb = (nnz_per_dev * 12 + n_per_dev * 4 * 8 + n_per_dev * 8) / 2^20
    @printf("  VRAM estimate/device: %.1f MB\n", vram_mb)

    # Warmup
    print("  Warmup... ")
    sync_all_devices(ndev)
    t_w = @elapsed begin
        mdla_solve(A_md, b_md; atol=1e-12, rtol=1e-12)
        sync_all_devices(ndev)
    end
    @printf("%.3f s\n", t_w)

    # Timed runs
    times = Float64[]
    local last_stats
    for run in 1:nruns
        sync_all_devices(ndev)
        t = @elapsed begin
            _, last_stats = mdla_solve(A_md, b_md; atol=1e-12, rtol=1e-12)
            sync_all_devices(ndev)
        end
        push!(times, t)
    end
    t_med = median(times)

    # Brief correctness check
    x_md, stats = mdla_solve(A_md, b_md; atol=1e-12, rtol=1e-12)
    y_md = similar(b_md)
    mul!(y_md, A_md, x_md)
    rel_res = norm(gather(y_md) - b_cpu) / norm(b_cpu)

    @printf("  Solve: %.4f s  (%d iters)  rel_residual=%.1e  converged=%s\n",
        t_med, last_stats.niter, rel_res, stats.solved)
    println()

    push!(results, (ndev, t_med, last_stats.niter, vram_mb))
end

# ── Summary table ─────────────────────────────────────────────────────────────

println("=" ^ 70)
println("Summary  ($(nx)×$(ny), $nruns runs)")
println("=" ^ 70)

t_base = results[1][2]  # 1-device as baseline

@printf("%-12s %10s %7s %10s %10s\n", "Devices", "Time (s)", "Iters", "VRAM/dev", "Speedup")
println("─" ^ 55)
for (ndev, t, iters, vram) in results
    spd = t_base / t
    @printf("%-12d %10.4f %7d %8.1f MB %9.2fx\n", ndev, t, iters, vram, spd)
end
println()
