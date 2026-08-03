using CUDA.CUSPARSE
using MultiDeviceLinearAlgebra: _assemble_from_blocks, _specs_match, DeviceIndex

# Rows and columns are partitioned independently, so a distributed matrix may be rectangular.
# Nothing here multiplies two sparse matrices yet — that is the next stage — but every one of
# these paths is one the triple product walks, and each has a failure mode that produces
# plausible wrong numbers rather than an error. `gather` in particular un-remaps columns, so a
# `gather` that used the row partition would quietly transpose a rectangular matrix's column
# space into nonsense.

"""
Split `A_cpu` into per-device CSR row blocks whose column indices are left **global**, which is
the form `_assemble_from_blocks` consumes and the form a distributed multiply produces.
"""
function _global_column_blocks(A_cpu::SparseMatrixCSC{Tv}, row_spec) where {Tv}
    At = SparseMatrixCSC(sparse(A_cpu'))
    ncols = size(A_cpu, 2)
    blocks = Vector{CuSparseMatrixCSR{Tv, DeviceIndex}}(undef, row_spec.ndevices)
    for d in 1:row_spec.ndevices
        CUDA.device!(device_id(row_spec, d))
        r = row_spec.ranges[d]
        s = At.colptr[first(r)]
        e = At.colptr[last(r) + 1] - 1
        blocks[d] = CuSparseMatrixCSR{Tv, DeviceIndex}(
            CuVector{DeviceIndex}(DeviceIndex.(At.colptr[first(r):(last(r) + 1)] .- (s - 1))),
            CuVector{DeviceIndex}(DeviceIndex.(At.rowval[s:e])),
            CuVector{Tv}(At.nzval[s:e]),
            (length(r), ncols),
        )
    end
    return blocks
end

@testset "Rectangular MultiDeviceSparseMatrixCSR" begin
    @testset "ndev=$ndev" for ndev in DEVICE_COUNTS
        @testset "Device index arrays are $DeviceIndex" begin
            # cuSPARSE's sparse-sparse routines are 32-bit-index only, so the device
            # representation is pinned regardless of the host matrix's index type. `gather`
            # still hands back host-native `Int`.
            A_cpu = sprand(Float64, 40, 40, 0.1) + 5.0 * sparse(I, 40, 40)
            @test eltype(A_cpu.rowval) === Int
            A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)

            @test eltype(A_md.partitions[1].rowPtr) === DeviceIndex
            @test eltype(A_md.partitions[1].colVal) === DeviceIndex
            @test eltype(gather(A_md).rowval) === Int
        end

        @testset "Square matrices share one spec" begin
            A_cpu = sprand(Float64, 40, 40, 0.1) + 5.0 * sparse(I, 40, 40)
            A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = ndev)
            # Identity, not just equality: a caller can tell a genuinely square matrix from a
            # rectangular one whose partitions happen to coincide.
            @test A_md.col_spec === A_md.row_spec
        end

        @testset "Gather round-trip, $label" for (label, m, n) in
            (("tall (m > n)", 60, 24), ("wide (m < n)", 24, 60), ("square", 40, 40))
            A_cpu = sprand(Float64, m, n, 0.15)
            row_spec = compute_partition_ranges(m, ndev)
            col_spec = compute_partition_ranges(n, ndev)
            A_md = MultiDeviceSparseMatrixCSR(A_cpu, row_spec, col_spec)

            @test size(A_md) == (m, n)
            @test size(A_md, 1) == m
            @test size(A_md, 2) == n
            @test _specs_match(A_md.row_spec, row_spec)
            @test _specs_match(A_md.col_spec, col_spec)
            @test gather(A_md) ≈ A_cpu
        end

        @testset "Rectangular SpMV" begin
            m, n = 60, 24
            A_cpu = sprand(Float64, m, n, 0.15)
            x_cpu = randn(n)
            y_expected = A_cpu * x_cpu

            row_spec = compute_partition_ranges(m, ndev)
            col_spec = compute_partition_ranges(n, ndev)
            A_md = MultiDeviceSparseMatrixCSR(A_cpu, row_spec, col_spec)

            # x lives in the column space, y in the row space — different partitions.
            x_md = MultiDeviceVector(x_cpu, col_spec)
            y_md = MultiDeviceVector(zeros(m), row_spec)

            mul!(y_md, A_md, x_md)
            @test gather(y_md) ≈ y_expected rtol = 1.0e-10

            # 5-arg form on the same shapes
            α, β = 2.5, -1.5
            y_cpu = randn(m)
            y_md2 = MultiDeviceVector(copy(y_cpu), row_spec)
            mul!(y_md2, A_md, x_md, α, β)
            @test gather(y_md2) ≈ β * y_cpu + α * y_expected rtol = 1.0e-10
        end

        @testset "Prolongation as a distributed operator" begin
            # The AMG fixture: an 8×8 grid coarsened 2×2, so P is 64×16. The coarse space is
            # the tight dimension — sized so the sweep still has a column per device on a host
            # with many GPUs.
            P_cpu = prolongation_matrix_2d(8, 8; bx = 2, by = 2)
            nf, nc = size(P_cpu)
            fine_spec = compute_partition_ranges(nf, ndev)
            coarse_spec = compute_partition_ranges(nc, ndev)
            P_md = MultiDeviceSparseMatrixCSR(P_cpu, fine_spec, coarse_spec)

            @test gather(P_md) ≈ P_cpu

            xc = randn(nc)
            x_md = MultiDeviceVector(xc, coarse_spec)
            y_md = MultiDeviceVector(zeros(nf), fine_spec)
            mul!(y_md, P_md, x_md)
            @test gather(y_md) ≈ P_cpu * xc rtol = 1.0e-10
        end

        @testset "_assemble_from_blocks round-trip, $label" for (label, m, n) in
            (("rectangular", 48, 20), ("square", 40, 40))
            A_cpu = sprand(Float64, m, n, 0.15)
            row_spec = compute_partition_ranges(m, ndev)
            col_spec = compute_partition_ranges(n, ndev)

            blocks = _global_column_blocks(A_cpu, row_spec)
            A_md = _assemble_from_blocks(blocks, row_spec, col_spec, (m, n))

            @test size(A_md) == (m, n)
            @test gather(A_md) ≈ A_cpu

            # The ghost columns it discovered must be sorted and disjoint from what each
            # device owns — the layout the exchange indexes into.
            for d in 1:ndev
                ghosts = A_md.ghost_exchange.ghost_global_indices[d]
                @test issorted(ghosts)
                @test all(!in(col_spec.ranges[d]), ghosts)
                @test all(g -> 1 <= g <= n, ghosts)
            end

            # An SpMV through the assembled exchange proves the halo it built actually works,
            # not merely that the blocks gathered back correctly.
            x_cpu = randn(n)
            x_md = MultiDeviceVector(x_cpu, col_spec)
            y_md = MultiDeviceVector(zeros(m), row_spec)
            mul!(y_md, A_md, x_md)
            @test gather(y_md) ≈ A_cpu * x_cpu rtol = 1.0e-10
        end
    end

    @testset "Error paths" begin
        ndev = min(2, NGPUS)

        @testset "Two-argument constructor rejects a rectangular matrix" begin
            A_cpu = sprand(Float64, 40, 20, 0.2)
            spec = compute_partition_ranges(40, ndev)
            @test_throws DimensionMismatch MultiDeviceSparseMatrixCSR(A_cpu, spec)
        end

        @testset "Specs must cover the matrix" begin
            A_cpu = sprand(Float64, 40, 20, 0.2)
            @test_throws DimensionMismatch MultiDeviceSparseMatrixCSR(
                A_cpu, compute_partition_ranges(39, ndev), compute_partition_ranges(20, ndev)
            )
            @test_throws DimensionMismatch MultiDeviceSparseMatrixCSR(
                A_cpu, compute_partition_ranges(40, ndev), compute_partition_ranges(21, ndev)
            )
        end

        @testset "Row and column partitions must agree on devices" begin
            if NGPUS >= 2
                A_cpu = sprand(Float64, 40, 20, 0.2)
                @test_throws ArgumentError MultiDeviceSparseMatrixCSR(
                    A_cpu,
                    compute_partition_ranges(40, 2; devices = [0, 1]),
                    compute_partition_ranges(20, 2; devices = [1, 0]),
                )
                @test_throws ArgumentError MultiDeviceSparseMatrixCSR(
                    A_cpu,
                    compute_partition_ranges(40, 2),
                    compute_partition_ranges(20, 1),
                )
            end
        end

        @testset "_assemble_from_blocks rejects disagreeing specs" begin
            if NGPUS >= 2
                # Same guardrail as the host constructor: a device-order mismatch between
                # the two specs would place partitions on a different device than the SpMV
                # runs on — unified addressing makes that run slowly instead of erroring.
                m, n = 40, 20
                A_cpu = sprand(Float64, m, n, 0.2)
                row_spec = compute_partition_ranges(m, 2)
                blocks = _global_column_blocks(A_cpu, row_spec)
                @test_throws ArgumentError _assemble_from_blocks(
                    blocks, row_spec,
                    compute_partition_ranges(n, 2; devices = [1, 0]), (m, n)
                )
                @test_throws ArgumentError _assemble_from_blocks(
                    blocks, row_spec, compute_partition_ranges(n, 1), (m, n)
                )
            end
        end

        @testset "SpMV rejects a mismatched vector partition" begin
            if NGPUS >= 2
                # Same total length, different split: a length-only check would accept this
                # and scatter every device's halo from the wrong entries.
                n = 40
                A_cpu = sprand(Float64, n, n, 0.15) + 5.0 * sparse(I, n, n)
                A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices = 2)
                skewed = PartitionSpec([1:10, 11:40])

                x_bad = MultiDeviceVector(randn(n), skewed)
                y_ok = MultiDeviceVector(zeros(n), A_md.row_spec)
                @test_throws ArgumentError mul!(y_ok, A_md, x_bad)

                x_ok = MultiDeviceVector(randn(n), A_md.col_spec)
                y_bad = MultiDeviceVector(zeros(n), skewed)
                @test_throws ArgumentError mul!(y_bad, A_md, x_ok)
            end
        end
    end
end
