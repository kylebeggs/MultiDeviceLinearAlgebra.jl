@testset "Krylov.jl CG integration" begin
    @testset "ndev=$ndev" for ndev in 1:min(NGPUS, 4)
        @testset "CG solve small SPD system" begin
            n = 200
            A_cpu = sprand(Float64, n, n, 0.05)
            A_cpu = A_cpu + A_cpu' + 20.0 * sparse(I, n, n)  # SPD
            x_true = randn(n)
            b_cpu = A_cpu * x_true

            A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices=ndev)
            b_md = MultiDeviceVector(b_cpu; ndevices=ndev)

            x_md, stats = Krylov.cg(A_md, b_md; atol=1e-12, rtol=1e-12)

            @test stats.solved
            x_result = gather(x_md)
            @test x_result ≈ x_true rtol=1e-6

            # Verify residual
            y_md = similar(b_md)
            mul!(y_md, A_md, x_md)
            r = gather(y_md) - b_cpu
            @test norm(r) / norm(b_cpu) < 1e-10
        end

        @testset "mdla_solve convenience function" begin
            n = 100
            A_cpu = sprand(Float64, n, n, 0.1)
            A_cpu = A_cpu + A_cpu' + 15.0 * sparse(I, n, n)
            x_true = randn(n)
            b_cpu = A_cpu * x_true

            A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices=ndev)
            b_md = MultiDeviceVector(b_cpu; ndevices=ndev)

            x_md, stats = mdla_solve(A_md, b_md; atol=1e-12, rtol=1e-12)
            @test stats.solved
            @test gather(x_md) ≈ x_true rtol=1e-6
        end
    end
end

@testset "LinearSolve.jl integration" begin
    using LinearSolve

    @testset "ndev=$ndev" for ndev in 1:min(NGPUS, 4)
        n = 100
        A_cpu = sprand(Float64, n, n, 0.1)
        A_cpu = A_cpu + A_cpu' + 15.0 * sparse(I, n, n)
        x_true = randn(n)
        b_cpu = A_cpu * x_true

        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices=ndev)
        b_md = MultiDeviceVector(b_cpu; ndevices=ndev)

        prob = LinearProblem(A_md, b_md)
        sol = solve(prob, KrylovJL_CG(); abstol=1e-12, reltol=1e-12)

        x_result = gather(sol.u)
        @test x_result ≈ x_true rtol=1e-6
    end
end

@testset "Iteration count consistency" begin
    n = 200
    A_cpu = sprand(Float64, n, n, 0.05)
    A_cpu = A_cpu + A_cpu' + 20.0 * sparse(I, n, n)
    x_true = randn(n)
    b_cpu = A_cpu * x_true

    iter_counts = Int[]
    for ndev in 1:min(NGPUS, 4)
        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices=ndev)
        b_md = MultiDeviceVector(b_cpu; ndevices=ndev)

        _, stats = Krylov.cg(A_md, b_md; atol=1e-12, rtol=1e-12)
        @test stats.solved
        push!(iter_counts, stats.niter)
    end

    @test allequal(iter_counts)
end
