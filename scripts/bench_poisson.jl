using MultiDeviceLinearAlgebra
using CUDA
using Krylov
using LinearAlgebra
using SparseArrays
using Printf

ngpus = length(CUDA.devices())
println("Available GPUs: $ngpus")
for d in CUDA.devices()
    println("  ", d, " — ", CUDA.name(d))
end
println()

# Problem size — adjust to taste
nx = ny = parse(Int, get(ENV, "POISSON_NX", "500"))
N = nx * ny
hx = 1.0 / (nx + 1)
hy = 1.0 / (ny + 1)

println("2D Poisson: $(nx)×$(ny) grid → $N unknowns")
println()

# Build matrix and RHS (manufactured solution u = sin(πx)sin(πy))
print("Assembling matrix... ")
t_assemble = @elapsed A_cpu = poisson_matrix_2d(nx, ny)
@printf("%.3f s  (nnz = %d)\n", t_assemble, nnz(A_cpu))

b_cpu = zeros(N)
u_exact = zeros(N)
for j in 1:ny, i in 1:nx
    x = i * hx
    y = j * hy
    idx = (j - 1) * nx + i
    b_cpu[idx] = 2π^2 * sin(π * x) * sin(π * y)
    u_exact[idx] = sin(π * x) * sin(π * y)
end

# Sweep over device counts: 1, 2, … , ngpus
for ndev in 1:ngpus
    println("─" ^ 60)
    println("Devices: $ndev")

    # Upload
    print("  Upload (CSC→CSR + vectors)... ")
    t_upload = @elapsed begin
        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices=ndev)
        b_md = MultiDeviceVector(b_cpu; ndevices=ndev)
    end
    @printf("%.3f s\n", t_upload)

    # Warmup solve (compilation)
    print("  Warmup solve... ")
    t_warmup = @elapsed begin
        x_md, _ = mdla_solve(A_md, b_md; atol=1e-12, rtol=1e-12)
    end
    @printf("%.3f s\n", t_warmup)

    # Timed solve
    CUDA.synchronize()
    print("  Timed solve...  ")
    t_solve = @elapsed begin
        x_md, stats = mdla_solve(A_md, b_md; atol=1e-12, rtol=1e-12)
        CUDA.synchronize()
    end
    @printf("%.4f s  (%d iterations)\n", t_solve, stats.niter)

    # Verify
    u_gpu = gather(x_md)
    fd_err = norm(u_gpu - u_exact, Inf)

    y_md = similar(b_md)
    mul!(y_md, A_md, x_md)
    rel_res = norm(gather(y_md) - b_cpu) / norm(b_cpu)

    @printf("  FD error (vs exact):   %.2e\n", fd_err)
    @printf("  Relative residual:     %.2e\n", rel_res)
    @printf("  Converged: %s\n", stats.solved)
    println()
end
