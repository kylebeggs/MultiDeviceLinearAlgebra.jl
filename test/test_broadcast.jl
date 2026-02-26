@testset "MultiDeviceVector broadcasting" begin
    n = 100
    ndev = min(NGPUS, 4)

    @testset "y .= α .* x" begin
        α = 2.5
        x_cpu = randn(n)
        x_md = MultiDeviceVector(x_cpu; ndevices=ndev)
        y_md = similar(x_md)
        y_md .= α .* x_md
        @test gather(y_md) ≈ α .* x_cpu rtol=1e-12
    end

    @testset "y .= x .+ z" begin
        x_cpu = randn(n)
        z_cpu = randn(n)
        x_md = MultiDeviceVector(x_cpu; ndevices=ndev)
        z_md = MultiDeviceVector(z_cpu; ndevices=ndev)
        y_md = similar(x_md)
        y_md .= x_md .+ z_md
        @test gather(y_md) ≈ x_cpu .+ z_cpu rtol=1e-12
    end

    @testset "y .= α .* x .+ β .* z" begin
        α = 2.0
        β = -0.5
        x_cpu = randn(n)
        z_cpu = randn(n)
        x_md = MultiDeviceVector(x_cpu; ndevices=ndev)
        z_md = MultiDeviceVector(z_cpu; ndevices=ndev)
        y_md = similar(x_md)
        y_md .= α .* x_md .+ β .* z_md
        @test gather(y_md) ≈ α .* x_cpu .+ β .* z_cpu rtol=1e-12
    end

    @testset "In-place y .= y .+ x" begin
        x_cpu = randn(n)
        y_cpu = randn(n)
        x_md = MultiDeviceVector(x_cpu; ndevices=ndev)
        y_md = MultiDeviceVector(copy(y_cpu); ndevices=ndev)
        y_md .= y_md .+ x_md
        @test gather(y_md) ≈ y_cpu .+ x_cpu rtol=1e-12
    end

    @testset "y .= x ./ s (scalar division)" begin
        s = 3.0
        x_cpu = randn(n)
        x_md = MultiDeviceVector(x_cpu; ndevices=ndev)
        y_md = similar(x_md)
        y_md .= x_md ./ s
        @test gather(y_md) ≈ x_cpu ./ s rtol=1e-12
    end
end
