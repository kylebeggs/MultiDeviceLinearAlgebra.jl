using MultiDeviceLinearAlgebra
using Test
using JET

@testset "MultiDeviceLinearAlgebra.jl" begin
    @testset "Code linting (JET.jl)" begin
        JET.test_package(MultiDeviceLinearAlgebra; target_defined_modules = true)
    end
    # Write your tests here.
end
