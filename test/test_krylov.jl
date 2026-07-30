@testset "Krylov.jl CG integration" begin
    @testset "ndev=$ndev" for ndev in DEVICE_COUNTS
        @testset "CG solve small SPD system" begin
            n = 200
            A_cpu = sprand(Float64, n, n, 0.05)
            A_cpu = A_cpu + A_cpu' + 20.0 * sparse(I, n, n)  # SPD
            x_true = randn(n)
            b_cpu = A_cpu * x_true

            A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
            b_md = MultiDeviceVector(b_cpu; ndevices = ndev)

            x_md, stats = Krylov.cg(A_md, b_md; atol = 1.0e-12, rtol = 1.0e-12)

            @test stats.solved
            x_result = gather(x_md)
            @test x_result ≈ x_true rtol = 1.0e-6

            # Verify residual
            y_md = similar(b_md)
            mul!(y_md, A_md, x_md)
            r = gather(y_md) - b_cpu
            @test norm(r) / norm(b_cpu) < 1.0e-10
        end

        @testset "mdla_solve convenience function" begin
            n = 100
            A_cpu = sprand(Float64, n, n, 0.1)
            A_cpu = A_cpu + A_cpu' + 15.0 * sparse(I, n, n)
            x_true = randn(n)
            b_cpu = A_cpu * x_true

            A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
            b_md = MultiDeviceVector(b_cpu; ndevices = ndev)

            x_md, stats = mdla_solve(A_md, b_md; atol = 1.0e-12, rtol = 1.0e-12)
            @test stats.solved
            @test gather(x_md) ≈ x_true rtol = 1.0e-6
        end
    end
end

@testset "LinearSolve.jl integration" begin
    using LinearSolve

    @testset "ndev=$ndev" for ndev in DEVICE_COUNTS
        n = 100
        A_cpu = sprand(Float64, n, n, 0.1)
        A_cpu = A_cpu + A_cpu' + 15.0 * sparse(I, n, n)
        x_true = randn(n)
        b_cpu = A_cpu * x_true

        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
        b_md = MultiDeviceVector(b_cpu; ndevices = ndev)

        prob = LinearProblem(A_md, b_md)
        sol = solve(prob, KrylovJL_CG(); abstol = 1.0e-12, reltol = 1.0e-12)

        x_result = gather(sol.u)
        @test x_result ≈ x_true rtol = 1.0e-6
    end
end

# Iteration count vs device count.
#
# This used to be a single `@test allequal(iter_counts)` on `A_rand + A_rand' + 20I` at n = 200 —
# a matrix so strongly diagonally dominant that CG finishes long before rounding has room to
# compound, so the test could not exercise the property it named. On a real Poisson system the
# assertion is simply false: partitioning changes cuSPARSE's CSR row-block schedule, and no
# host-side change can make the result partition-invariant (see the README's "Reproducibility
# across device counts"). Issue #27.
#
# It is split into the two claims that *are* defensible: exact equality where the arithmetic
# guarantees it, and agreement of the actual answer everywhere else.

@testset "Iteration count on exactly-solvable systems" begin
    # Small-integer entries and one DOF per device make the arithmetic exact, so CG stops on its
    # finite-termination property rather than on a noise-driven residual. `_dense_spd` is the
    # decisive case: its spectrum is just `{n, 2n}`, so exact CG terminates in two steps however
    # the rows are split. Equality here is a real invariant, not a hopeful one.
    @testset "$label" for (label, build) in
        (("tridiagonal", _tridiagonal_spd), ("dense", _dense_spd))

        n = maximum(DEVICE_COUNTS)
        if n >= 2
            A_cpu = build(n)
            b_cpu = collect(1.0:n)

            iter_counts = Int[]
            for ndev in DEVICE_COUNTS
                A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
                spec = A_md.row_spec
                b_md = MultiDeviceVector(b_cpu, spec)

                x_md, stats = mdla_solve(A_md, b_md; atol = 1.0e-12, rtol = 1.0e-12)
                _sync_devices(spec)
                @test stats.solved

                y_md = similar(b_md)
                mul!(y_md, A_md, x_md)
                _sync_devices(spec)
                @test norm(gather(y_md) - b_cpu) / norm(b_cpu) < 1.0e-12

                push!(iter_counts, stats.niter)
            end

            @test allequal(iter_counts)
        end
    end
end

@testset "Iteration count and solution agreement on a well-posed Poisson system" begin
    # Well posed in the sense that matters: `rtol` sits above the attainable residual floor
    # `ε‖A‖‖x‖/‖b‖` (~4e-12 here), so CG converges on the true residual instead of stopping when
    # its recursively updated one drifts under an unreachable threshold. A fixed-seed random RHS
    # avoids the analytic sin·sin forcing, which is an exact eigenvector of this matrix.
    nx = 200
    A_cpu = poisson_matrix_2d(nx, nx)
    Random.seed!(0x00C0FFEE)
    b_cpu = randn(nx * nx)
    normb = norm(b_cpu)

    iter_counts = Int[]
    solutions = Vector{Float64}[]
    for ndev in DEVICE_COUNTS
        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
        spec = A_md.row_spec
        b_md = MultiDeviceVector(b_cpu, spec)

        x_md, stats = mdla_solve(A_md, b_md; atol = 0.0, rtol = 1.0e-8)
        _sync_devices(spec)
        @test stats.solved

        y_md = similar(b_md)
        mul!(y_md, A_md, x_md)
        _sync_devices(spec)
        @test norm(gather(y_md) - b_cpu) / normb < 1.0e-7

        push!(iter_counts, stats.niter)
        push!(solutions, gather(x_md))
    end

    # A band, not equality. Measured on 9×A30 the count was identical across 1/2/4/9 devices at
    # both 1000² and 2000² Poisson, but that is an observation about one host's CUBLAS/cuSPARSE
    # and not something the package can guarantee — so the guard allows drift while still
    # failing on the kind of blow-up a broken partition would cause.
    lo, hi = extrema(iter_counts)
    @test hi - lo <= max(2, ceil(Int, 0.05 * lo))

    # The assertion that actually catches a broken partition: same problem, same answer,
    # regardless of how it was split. Independent of any rounding-driven iteration count.
    for k in 2:length(solutions)
        @test solutions[k] ≈ solutions[1] rtol = 1.0e-6
    end
end
