using MultiDeviceLinearAlgebra
using CUDA
using LinearAlgebra
using Printf

nx = ny = parse(Int, get(ENV, "POISSON_NX", "100"))
N = nx * ny
hx = 1.0 / (nx + 1)
hy = 1.0 / (ny + 1)

println("2D Poisson correctness check: $(nx)×$(ny) grid → $N unknowns")

A_cpu = poisson_matrix_2d(nx, ny)

b_cpu = zeros(N)
u_exact = zeros(N)
for j in 1:ny, i in 1:nx
    x = i * hx
    y = j * hy
    idx = (j - 1) * nx + i
    b_cpu[idx] = 2π^2 * sin(π * x) * sin(π * y)
    u_exact[idx] = sin(π * x) * sin(π * y)
end

ngpus = length(CUDA.devices())
@assert CUDA.functional() && ngpus >= 1 "Need at least 1 CUDA GPU"
println("GPUs available: $ngpus")
println()

all_passed = true

for ndev in 1:ngpus
    A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
    b_md = MultiDeviceVector(b_cpu; ndevices = ndev)

    x_md, stats = mdla_solve(A_md, b_md; atol = 1.0e-12, rtol = 1.0e-12)

    u_num = gather(x_md)
    fd_err = norm(u_num - u_exact, Inf)

    y_md = similar(b_md)
    mul!(y_md, A_md, x_md)
    rel_res = norm(gather(y_md) - b_cpu) / norm(b_cpu)

    converged = stats.solved
    passed = converged && rel_res < 1.0e-8

    status = passed ? "PASS" : "FAIL"
    @printf(
        "  %d GPU(s): [%s]  converged=%s  fd_error=%.2e  rel_residual=%.2e  iters=%d\n",
        ndev, status, converged, fd_err, rel_res, stats.niter
    )

    if !passed
        global all_passed = false
    end
end

println()
if all_passed
    println("All checks passed.")
else
    println("Some checks FAILED.")
    exit(1)
end
