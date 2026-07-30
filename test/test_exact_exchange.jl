# Exactness and determinism of the halo exchange, with no rounding to hide behind.
#
# Every other SpMV test compares with `≈`, so a communication fault small enough to sit
# inside the tolerance passes unnoticed — and on a problem with real rounding there is no
# way to tell a stale ghost from re-association. These testsets remove the ambiguity by
# choosing systems whose arithmetic is *exact* in Float64, which makes `==` the correct
# comparison and any mismatch attributable to communication alone.
#
# Promoted from `scripts/diagnose_partition_sensitivity.jl` §1/§2 after that diagnostic
# cleared issue #27 on the 9×A30 host. The determinism testset is also the regression test
# asked for in issue #30.

# `_tridiagonal_spd`, `_dense_spd` and `_sync_devices` are shared helpers from
# `test/runtests.jl`; `test/test_krylov.jl` uses the same matrix builders.

@testset "Exact halo exchange" begin
    # n = ndevices puts exactly one row on each device, so every off-diagonal entry is a
    # ghost: the highest communication-to-compute ratio the package can be given.
    @testset "one DOF per device, $label, ndev=$ndev" for ndev in DEVICE_COUNTS,
            (label, build) in (("tridiagonal", _tridiagonal_spd), ("dense", _dense_spd))

        # A dense n×n at n = 1 is the tridiagonal, and there is nothing to exchange.
        if !(label == "dense" && ndev < 2)
            n = ndev
            A_cpu = build(n)
            x_cpu = collect(1.0:n)
            y_exact = A_cpu * x_cpu

            A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
            spec = A_md.row_spec
            x_md = MultiDeviceVector(x_cpu, spec)
            y_md = MultiDeviceVector(zeros(n), spec)

            mul!(y_md, A_md, x_md)
            _sync_devices(spec)

            # Bitwise, not `≈`: this arithmetic has no rounding, so any difference is
            # communication. Ghosts arriving as zero would leave the diagonal term alone.
            @test gather(y_md) == y_exact
        end
    end

    @testset "2×2 from the issue #27 thread" begin
        if NGPUS >= 2
            # Spelled out so the guard stands on its own: one DOF per GPU, one value
            # exchanged. Correct is [2, 7]; ghosts arriving as zero would give [4, 8].
            A_cpu = _tridiagonal_spd(2)
            @test Array(A_cpu) == [4.0 -1.0; -1.0 4.0]

            A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = 2)
            spec = A_md.row_spec
            x_md = MultiDeviceVector([1.0, 2.0], spec)
            y_md = MultiDeviceVector(zeros(2), spec)

            mul!(y_md, A_md, x_md)
            _sync_devices(spec)

            @test gather(y_md) == [2.0, 7.0]
        end
    end

    # CG issues `mul!` after `mul!` with nothing draining the devices in between, and every
    # call reuses the same send/recv/`local_x` buffers. A stale-buffer race lives only in
    # that window — every other testset gathers after each call, inserting the very
    # synchronization that would hide it. This is the regression test issue #30 asks for.
    @testset "back-to-back SpMV without host sync, ndev=$ndev" for ndev in DEVICE_COUNTS
        if ndev >= 2
            repeats = 50
            n = ndev
            A_cpu = _dense_spd(n)                       # every ordered pair under load
            xa_cpu = collect(1.0:n)
            xb_cpu = collect(Float64, n:-1:1)
            ya, yb = A_cpu * xa_cpu, A_cpu * xb_cpu

            A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
            spec = A_md.row_spec
            xa = MultiDeviceVector(xa_cpu, spec)
            xb = MultiDeviceVector(xb_cpu, spec)
            ys = [MultiDeviceVector(zeros(n), spec) for _ in 1:repeats]

            # Alternating inputs are what makes staleness visible: a call that reads the
            # previous iteration's send buffer produces the *other* vector's answer, and
            # the two differ in every entry.
            for k in 1:repeats
                mul!(ys[k], A_md, isodd(k) ? xa : xb)
            end
            _sync_devices(spec)

            @test all(k -> gather(ys[k]) == (isodd(k) ? ya : yb), 1:repeats)
        end
    end

    # Real halo, real sparsity. The arithmetic is no longer exact, so the assertion is
    # self-consistency: identical input through the same operator must give identical bits
    # every time. Any variation at all is nondeterminism, whatever its source.
    @testset "repeated Poisson SpMV is bitwise self-consistent, ndev=$ndev" for
        ndev in DEVICE_COUNTS

        if ndev >= 2
            nx = 200
            repeats = 10
            A_cpu = poisson_matrix_2d(nx, nx)
            Random.seed!(0x00C0FFEE)
            x_cpu = randn(nx * nx)

            A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
            spec = A_md.row_spec
            x_md = MultiDeviceVector(x_cpu, spec)
            ys = [MultiDeviceVector(zeros(nx * nx), spec) for _ in 1:repeats]

            for k in 1:repeats
                mul!(ys[k], A_md, x_md)
            end
            _sync_devices(spec)

            ref = gather(ys[1])
            @test all(k -> gather(ys[k]) == ref, 2:repeats)
        end
    end
end
