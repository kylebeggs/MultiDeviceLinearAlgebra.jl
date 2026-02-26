@testset "MultiDeviceSparseMatrixCSR" begin
    ndev = min(NGPUS, 4)

    @testset "Construction and size" begin
        n = 50
        A_cpu = sprand(Float64, n, n, 0.1) + 5.0 * sparse(I, n, n)
        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices=ndev)

        @test size(A_md) == (n, n)
        @test size(A_md, 1) == n
        @test size(A_md, 2) == n
        @test eltype(A_md) == Float64
    end

    @testset "Gather round-trip" begin
        n = 50
        A_cpu = sprand(Float64, n, n, 0.1) + 5.0 * sparse(I, n, n)
        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices=ndev)
        A_back = gather(A_md)
        @test A_back ≈ A_cpu
    end

    @testset "SpMV mul! correctness" begin
        n = 200
        A_cpu = sprand(Float64, n, n, 0.05) + 10.0 * sparse(I, n, n)
        x_cpu = randn(n)
        y_expected = A_cpu * x_cpu

        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices=ndev)
        x_md = MultiDeviceVector(x_cpu; ndevices=ndev)
        y_md = MultiDeviceVector(zeros(n); ndevices=ndev)

        mul!(y_md, A_md, x_md)
        y_result = gather(y_md)

        @test y_result ≈ y_expected rtol=1e-10
    end

    @testset "SpMV mul! with single device" begin
        n = 100
        A_cpu = sprand(Float64, n, n, 0.1) + 5.0 * sparse(I, n, n)
        x_cpu = randn(n)
        y_expected = A_cpu * x_cpu

        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices=1)
        x_md = MultiDeviceVector(x_cpu; ndevices=1)
        y_md = MultiDeviceVector(zeros(n); ndevices=1)

        mul!(y_md, A_md, x_md)
        @test gather(y_md) ≈ y_expected rtol=1e-10
    end

    @testset "5-arg mul! (α, β)" begin
        n = 100
        A_cpu = sprand(Float64, n, n, 0.1) + 5.0 * sparse(I, n, n)
        x_cpu = randn(n)
        y_cpu = randn(n)
        α = 2.0
        β = 0.5
        y_expected = β * y_cpu + α * A_cpu * x_cpu

        A_md = MultiDeviceSparseMatrixCSR(A_cpu; ndevices=ndev)
        x_md = MultiDeviceVector(x_cpu; ndevices=ndev)
        y_md = MultiDeviceVector(copy(y_cpu); ndevices=ndev)

        mul!(y_md, A_md, x_md, α, β)
        @test gather(y_md) ≈ y_expected rtol=1e-10
    end
end
