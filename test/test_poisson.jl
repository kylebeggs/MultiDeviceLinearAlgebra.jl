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

@testset "Prolongation matrix construction" begin
    @testset "Injection structure" begin
        P = prolongation_matrix_2d(4, 4; bx = 2, by = 2)

        @test size(P) == (16, 4)
        @test nnz(P) == 16                            # exactly one nonzero per fine point
        @test all(==(1), sum(P .!= 0, dims = 2))
        @test all(==(1.0), P.nzval)                   # unit weights
        @test vec(sum(P, dims = 1)) == fill(4.0, 4)   # each 2×2 aggregate has 4 members
    end

    @testset "Both indexings are column-major" begin
        # `poisson_matrix_2d` numbers fine point (i,j) as (j-1)*nx + i. The coarse space must
        # follow the same convention, or a contiguous block of fine unknowns stops mapping to
        # a contiguous block of coarse ones and the distributed halo degenerates from a
        # surface into an all-to-all.
        nx, ny, bx, by = 6, 6, 2, 2
        nxc = cld(nx, bx)
        P = prolongation_matrix_2d(nx, ny; bx = bx, by = by)

        for j in 1:ny, i in 1:nx
            fine = (j - 1) * nx + i
            coarse = (cld(j, by) - 1) * nxc + cld(i, bx)
            @test P[fine, coarse] == 1.0
        end
    end

    @testset "Partial trailing aggregates" begin
        # 7 = 3+3+1 columns of aggregates and 5 = 2+2+1 rows: the last block in each
        # direction is short, and every fine point must still be assigned exactly once.
        P = prolongation_matrix_2d(7, 5; bx = 3, by = 2)

        @test size(P) == (35, 9)
        @test nnz(P) == 35
        @test all(==(1), sum(P .!= 0, dims = 2))
        @test sum(P) == 35.0
        # Aggregate sizes: 3×2, 3×2, 1×2 across, and a final row of height 1.
        @test sort(vec(sum(P, dims = 1))) == [1.0, 2.0, 2.0, 3.0, 3.0, 6.0, 6.0, 6.0, 6.0]
    end

    @testset "Galerkin operator on the host" begin
        A = poisson_matrix_2d(4, 4)
        P = prolongation_matrix_2d(4, 4; bx = 2, by = 2)
        RAP = SparseMatrixCSC(P') * A * P

        @test size(RAP) == (4, 4)
        @test RAP ≈ RAP'          # symmetric because A is
        @test rank(Matrix(RAP)) == 4
    end

    @testset "Validation" begin
        @test_throws AssertionError prolongation_matrix_2d(0, 4)
        @test_throws AssertionError prolongation_matrix_2d(4, 4; bx = 0)
    end
end

if HAS_CUDA && NGPUS >= 1
    @testset "Poisson GPU solve" begin
        @testset "ndev=$ndev" for ndev in DEVICE_COUNTS
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

            A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
            b_md = MultiDeviceVector(b_cpu; ndevices = ndev)

            x_md, stats = mdla_solve(A_md, b_md; atol = 1.0e-12, rtol = 1.0e-12)
            @test stats.solved

            u_gpu = gather(x_md)
            @test norm(u_gpu - u_exact, Inf) < 0.05

            # Verify residual
            y_md = similar(b_md)
            mul!(y_md, A_md, x_md)
            residual = norm(gather(y_md) - b_cpu) / norm(b_cpu)
            @test residual < 1.0e-10
        end
    end
end
