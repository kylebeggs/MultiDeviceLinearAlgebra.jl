@testset "PartitionSpec & compute_partition_ranges" begin
    @testset "Even division" begin
        spec = compute_partition_ranges(100, 4)
        @test spec.len == 100
        @test spec.ndevices == 4
        @test spec.ranges == [1:25, 26:50, 51:75, 76:100]
        @test sum(length.(spec.ranges)) == 100
    end

    @testset "Uneven division" begin
        spec = compute_partition_ranges(103, 4)
        @test spec.len == 103
        @test spec.ndevices == 4
        @test length(spec.ranges[1]) == 26
        @test length(spec.ranges[2]) == 26
        @test length(spec.ranges[3]) == 26
        @test length(spec.ranges[4]) == 25
        @test sum(length.(spec.ranges)) == 103
        @test first(spec.ranges[1]) == 1
        @test last(spec.ranges[4]) == 103
    end

    @testset "Single device" begin
        spec = compute_partition_ranges(100, 1)
        @test spec.ranges == [1:100]
        @test spec.ndevices == 1
    end

    @testset "n == ndevices" begin
        spec = compute_partition_ranges(4, 4)
        @test all(r -> length(r) == 1, spec.ranges)
        @test spec.ranges == [1:1, 2:2, 3:3, 4:4]
    end

    @testset "Validation errors" begin
        @test_throws AssertionError compute_partition_ranges(0, 4)
        @test_throws AssertionError compute_partition_ranges(100, 0)
        @test_throws AssertionError compute_partition_ranges(3, 5)
    end

    @testset "Contiguous coverage" begin
        for (n, nd) in [(10, 3), (17, 5), (1000, 7), (1, 1)]
            spec = compute_partition_ranges(n, nd)
            @test sum(length.(spec.ranges)) == n
            @test first(spec.ranges[1]) == 1
            @test last(spec.ranges[end]) == n
            for i in 1:(nd - 1)
                @test last(spec.ranges[i]) + 1 == first(spec.ranges[i + 1])
            end
        end
    end
end

@testset "Manual PartitionSpec construction" begin
    @testset "Valid custom ranges" begin
        spec = PartitionSpec([1:10, 11:30, 31:100])
        @test spec.len == 100
        @test spec.ndevices == 3
        @test spec.ranges == [1:10, 11:30, 31:100]
    end

    @testset "Single range" begin
        spec = PartitionSpec([1:50])
        @test spec.len == 50
        @test spec.ndevices == 1
    end

    @testset "Empty ranges vector" begin
        @test_throws ArgumentError PartitionSpec(UnitRange{Int}[])
    end

    @testset "Empty individual range" begin
        @test_throws ArgumentError PartitionSpec([1:0, 1:10])
    end

    @testset "Not starting at 1" begin
        @test_throws ArgumentError PartitionSpec([2:10, 11:20])
    end

    @testset "Gap between ranges" begin
        @test_throws ArgumentError PartitionSpec([1:10, 12:20])
    end

    @testset "Overlapping ranges" begin
        @test_throws ArgumentError PartitionSpec([1:10, 10:20])
    end

    @testset "Accepts AbstractVector and integer subtypes" begin
        spec = PartitionSpec(UnitRange{Int32}[Int32(1):Int32(10), Int32(11):Int32(20)])
        @test spec.len == 20
        @test spec.ndevices == 2
        @test spec.ranges == [1:10, 11:20]
        @test eltype(spec.ranges) == UnitRange{Int}
    end
end

@testset "device_for_index" begin
    spec = compute_partition_ranges(10, 3)

    @test device_for_index(spec, 1) == (1, 1)
    @test device_for_index(spec, 4) == (1, 4)
    @test device_for_index(spec, 5) == (2, 1)
    @test device_for_index(spec, 7) == (2, 3)
    @test device_for_index(spec, 8) == (3, 1)
    @test device_for_index(spec, 10) == (3, 3)

    @test_throws BoundsError device_for_index(spec, 0)
    @test_throws BoundsError device_for_index(spec, 11)
end

@testset "device_id and devices field" begin
    @testset "Default devices (0-indexed)" begin
        spec = compute_partition_ranges(100, 4)
        @test spec.devices == 0:3
        @test spec.devices isa UnitRange{Int}
        for d in 1:4
            @test device_id(spec, d) == d - 1
        end
    end

    @testset "Default devices via manual PartitionSpec" begin
        spec = PartitionSpec([1:10, 11:20, 21:30])
        @test spec.devices == 0:2
        @test spec.devices isa UnitRange{Int}
    end

    @testset "Custom devices via compute_partition_ranges" begin
        spec = compute_partition_ranges(100, 3; devices = [3, 4, 5])
        @test spec.devices == [3, 4, 5]
        @test spec.devices isa Vector{Int}
        @test device_id(spec, 1) == 3
        @test device_id(spec, 2) == 4
        @test device_id(spec, 3) == 5
    end

    @testset "Custom devices via manual PartitionSpec" begin
        spec = PartitionSpec([1:50, 51:100]; devices = [2, 7])
        @test spec.devices == [2, 7]
        @test device_id(spec, 1) == 2
        @test device_id(spec, 2) == 7
    end

    @testset "Infer ndevices from devices" begin
        spec = compute_partition_ranges(100; devices = [1, 3, 5])
        @test spec.ndevices == 3
        @test spec.devices == [1, 3, 5]
        @test spec.len == 100
    end

    @testset "Validation: wrong length" begin
        @test_throws ArgumentError compute_partition_ranges(100, 3; devices = [0, 1])
        @test_throws ArgumentError PartitionSpec([1:50, 51:100]; devices = [0, 1, 2])
    end

    @testset "Validation: negative device IDs" begin
        @test_throws ArgumentError compute_partition_ranges(100, 2; devices = [-1, 0])
        @test_throws ArgumentError PartitionSpec([1:50, 51:100]; devices = [0, -1])
    end

    @testset "Validation: duplicate device IDs" begin
        @test_throws ArgumentError compute_partition_ranges(100, 3; devices = [0, 1, 1])
        @test_throws ArgumentError PartitionSpec([1:50, 51:100]; devices = [3, 3])
    end
end
