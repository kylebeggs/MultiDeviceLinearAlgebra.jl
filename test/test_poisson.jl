@testset "Poisson matrix construction" begin
    @testset "2D 5-point stencil structure" begin
        nx, ny = 4, 4
        A = poisson_matrix_2d(nx, ny)
        N = nx * ny

        @test size(A) == (N, N)
        @test issymmetric(A)
        @test all(eigvals(Matrix(A)) .> 0)  # SPD

        hx = 1.0 / (nx + 1)
        hy = 1.0 / (ny + 1)
        expected_diag = 2.0 / hx^2 + 2.0 / hy^2
        @test A[1, 1] ≈ expected_diag
    end

    @testset "Manufactured solution convergence" begin
        # Solve -Δu = f on [0,1]² with u=0 boundary
        # Manufactured: u(x,y) = sin(πx)sin(πy), f = 2π²sin(πx)sin(πy)
        nx = ny = 20
        hx = 1.0 / (nx + 1)
        hy = 1.0 / (ny + 1)
        N = nx * ny

        A_cpu = poisson_matrix_2d(nx, ny)

        # RHS from manufactured solution
        b_cpu = zeros(N)
        u_exact = zeros(N)
        for j in 1:ny, i in 1:nx
            x = i * hx
            y = j * hy
            idx = (j - 1) * nx + i
            b_cpu[idx] = 2 * π^2 * sin(π * x) * sin(π * y)
            u_exact[idx] = sin(π * x) * sin(π * y)
        end

        # Solve on CPU first to verify problem setup
        u_cpu = A_cpu \ b_cpu
        @test norm(u_cpu - u_exact, Inf) < 0.05  # FD error is O(h²)
    end
end

if HAS_CUDA && NGPUS >= 1
    @testset "Poisson GPU solve" begin
        @testset "ndev=$ndev" for ndev in 1:min(NGPUS, 4)
            nx = ny = 30
            hx = 1.0 / (nx + 1)
            hy = 1.0 / (ny + 1)
            N = nx * ny

            A_cpu = poisson_matrix_2d(nx, ny)

            b_cpu = zeros(N)
            u_exact = zeros(N)
            for j in 1:ny, i in 1:nx
                x = i * hx
                y = j * hy
                idx = (j - 1) * nx + i
                b_cpu[idx] = 2 * π^2 * sin(π * x) * sin(π * y)
                u_exact[idx] = sin(π * x) * sin(π * y)
            end

            A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices=ndev)
            b_md = MultiDeviceVector(b_cpu; ndevices=ndev)

            x_md, stats = mdla_solve(A_md, b_md; atol=1e-12, rtol=1e-12)
            @test stats.solved

            u_gpu = gather(x_md)
            @test norm(u_gpu - u_exact, Inf) < 0.05

            # Verify residual
            y_md = similar(b_md)
            mul!(y_md, A_md, x_md)
            residual = norm(gather(y_md) - b_cpu) / norm(b_cpu)
            @test residual < 1e-10
        end
    end
end
