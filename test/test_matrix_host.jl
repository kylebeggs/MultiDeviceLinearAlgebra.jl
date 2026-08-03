using MultiDeviceLinearAlgebra: _check_device_index_range, _check_spec_device_agreement,
    _revalidate_spec, _specs_match, DeviceIndex

# Host-side validation logic from src/matrix.jl. Everything here throws (or returns) before
# the first CUDA call, so this file runs in the CPU tier — it must never construct a matrix
# that survives validation, because the happy path immediately allocates on device.
#
# String-form @test_throws is deliberate: several functions have more than one throw branch,
# and matching the message is what tells a failing test WHICH branch misfired.
@testset "Host-side matrix validation" begin
    @testset "_check_spec_device_agreement" begin
        row_spec = compute_partition_ranges(60, 2)
        col_spec = compute_partition_ranges(40, 2)
        # A rectangular pairing on the same devices is the accepted case
        @test _check_spec_device_agreement(row_spec, col_spec) === nothing
        @test _check_spec_device_agreement(row_spec, row_spec) === nothing
        # Branch 1: device-count mismatch
        @test_throws "Row partition spans 2 devices but column partition spans 3" _check_spec_device_agreement(
            row_spec, compute_partition_ranges(60, 3)
        )
        # Branch 2: same count, different order — the case that would otherwise run slowly
        # over cross-device memory rather than error
        @test_throws "same CUDA devices in the same order" _check_spec_device_agreement(
            row_spec, compute_partition_ranges(40, 2; devices = [1, 0])
        )
    end

    @testset "_revalidate_spec" begin
        spec = PartitionSpec([1:10, 11:30, 31:100])
        @test _specs_match(_revalidate_spec(spec, "row"), spec)
        # The unchecked inner constructor can smuggle in an ndevices that lies about
        # length(ranges); revalidation must catch it
        bad_count = PartitionSpec([1:5, 6:10], 10, 3, [0, 1])
        @test_throws "Inconsistent row PartitionSpec" _revalidate_spec(bad_count, "row")
        # Non-contiguous ranges smuggled through the inner constructor are re-rejected by
        # the validating constructor _revalidate_spec runs
        bad_gap = PartitionSpec([1:5, 7:10], 10, 2, [0, 1])
        @test_throws "not contiguous" _revalidate_spec(bad_gap, "row")
    end

    @testset "_check_device_index_range" begin
        limit = Int(typemax(DeviceIndex))
        spec1 = compute_partition_ranges(1, 1)
        # Column count beyond Int32: a synthetic rowptr keeps this allocation-free
        @test_throws "may have at most" _check_device_index_range(Int64[1, 1], spec1, limit + 1)
        # Block nnz == typemax(Int32) overflows the device rowptr (the check is strict <)
        @test_throws "row pointers overflow" _check_device_index_range(
            Int64[1, 1 + limit], spec1, 10
        )
        # Boundary pass: limit - 1 nonzeros is the largest legal block
        @test _check_device_index_range(Int64[1, limit], spec1, 10) === nothing
        # Multi-device: the error names the offending device, not just "somewhere"
        spec2 = compute_partition_ranges(2, 2)
        @test_throws "Device 2 would hold" _check_device_index_range(
            Int64[1, 5, 5 + limit], spec2, 10
        )
    end

    @testset "Constructor validation precedes device work" begin
        A = sprand(10, 10, 0.3)
        # The square-only convenience constructor rejects a rectangular matrix
        @test_throws DimensionMismatch MultiDeviceSparseMatrixCSR(
            sprand(6, 4, 0.5), compute_partition_ranges(6, 2)
        )
        @test_throws "Row PartitionSpec covers 8 rows" MultiDeviceSparseMatrixCSR(
            A, compute_partition_ranges(8, 2), compute_partition_ranges(10, 2)
        )
        @test_throws "Column PartitionSpec covers 8 columns" MultiDeviceSparseMatrixCSR(
            A, compute_partition_ranges(10, 2), compute_partition_ranges(8, 2)
        )
        @test_throws "same CUDA devices in the same order" MultiDeviceSparseMatrixCSR(
            A, compute_partition_ranges(10, 2),
            compute_partition_ranges(10, 2; devices = [1, 0])
        )
    end
end
