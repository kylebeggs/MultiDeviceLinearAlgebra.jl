@testset "MultiDeviceVector" begin
    n = 100
    ndev = min(NGPUS, 4)

    @testset "Construction and gather round-trip" begin
        v_cpu = randn(Float64, n)
        v_md = MultiDeviceVector(v_cpu; ndevices=ndev)

        @test length(v_md) == n
        @test size(v_md) == (n,)
        @test eltype(v_md) == Float64
        @test v_md.spec.ndevices == ndev

        v_back = gather(v_md)
        @test v_back ≈ v_cpu
    end

    @testset "Construction with PartitionSpec" begin
        spec = compute_partition_ranges(n, ndev)
        v_cpu = randn(Float64, n)
        v_md = MultiDeviceVector(v_cpu, spec)
        @test gather(v_md) ≈ v_cpu
    end

    @testset "Undef constructor" begin
        spec = compute_partition_ranges(n, ndev)
        v_md = MultiDeviceVector{Float64}(undef, spec)
        @test length(v_md) == n
        @test v_md.spec.ndevices == ndev
    end

    @testset "similar preserves spec" begin
        v_cpu = randn(Float64, n)
        v_md = MultiDeviceVector(v_cpu; ndevices=ndev)
        w = similar(v_md)
        @test length(w) == n
        @test w.spec.ndevices == ndev
        @test w.spec.ranges == v_md.spec.ranges
        @test eltype(w) == Float64
    end

    @testset "similar with different type" begin
        v_md = MultiDeviceVector(randn(Float64, n); ndevices=ndev)
        w = similar(v_md, Float32)
        @test eltype(w) == Float32
        @test w.spec.ranges == v_md.spec.ranges
    end

    @testset "Scalar getindex/setindex!" begin
        v_cpu = collect(1.0:10.0)
        v_md = MultiDeviceVector(v_cpu; ndevices=ndev)
        for i in 1:10
            @test v_md[i] ≈ Float64(i)
        end
        v_md[5] = 99.0
        @test v_md[5] ≈ 99.0
    end

    @testset "fill!" begin
        v_md = MultiDeviceVector(zeros(n); ndevices=ndev)
        fill!(v_md, 3.14)
        result = gather(v_md)
        @test all(x -> x ≈ 3.14, result)
    end

    @testset "copyto!" begin
        a_cpu = randn(n)
        a_md = MultiDeviceVector(a_cpu; ndevices=ndev)
        b_md = similar(a_md)
        copyto!(b_md, a_md)
        @test gather(b_md) ≈ a_cpu
    end

    @testset "zero" begin
        v_md = MultiDeviceVector(randn(n); ndevices=ndev)
        z = zero(v_md)
        @test all(x -> x == 0.0, gather(z))
        @test z.spec.ranges == v_md.spec.ranges
    end

    @testset "dot" begin
        x_cpu = randn(n)
        y_cpu = randn(n)
        x_md = MultiDeviceVector(x_cpu; ndevices=ndev)
        y_md = MultiDeviceVector(y_cpu; ndevices=ndev)
        @test dot(x_md, y_md) ≈ dot(x_cpu, y_cpu) rtol=1e-12
    end

    @testset "norm" begin
        v_cpu = randn(n)
        v_md = MultiDeviceVector(v_cpu; ndevices=ndev)
        @test norm(v_md) ≈ norm(v_cpu) rtol=1e-12
    end

    @testset "axpy!" begin
        α = 2.5
        x_cpu = randn(n)
        y_cpu = randn(n)
        expected = y_cpu + α * x_cpu

        x_md = MultiDeviceVector(x_cpu; ndevices=ndev)
        y_md = MultiDeviceVector(copy(y_cpu); ndevices=ndev)
        axpy!(α, x_md, y_md)
        @test gather(y_md) ≈ expected rtol=1e-12
    end

    @testset "axpby!" begin
        α = 2.5
        β = -0.3
        x_cpu = randn(n)
        y_cpu = randn(n)
        expected = α * x_cpu + β * y_cpu

        x_md = MultiDeviceVector(x_cpu; ndevices=ndev)
        y_md = MultiDeviceVector(copy(y_cpu); ndevices=ndev)
        axpby!(α, x_md, β, y_md)
        @test gather(y_md) ≈ expected rtol=1e-12
    end

    @testset "rmul!" begin
        s = 3.7
        v_cpu = randn(n)
        expected = s * v_cpu
        v_md = MultiDeviceVector(copy(v_cpu); ndevices=ndev)
        rmul!(v_md, s)
        @test gather(v_md) ≈ expected rtol=1e-12
    end

    @testset "lmul!" begin
        s = -1.2
        v_cpu = randn(n)
        expected = s * v_cpu
        v_md = MultiDeviceVector(copy(v_cpu); ndevices=ndev)
        lmul!(s, v_md)
        @test gather(v_md) ≈ expected rtol=1e-12
    end
end
