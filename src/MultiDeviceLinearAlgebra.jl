module MultiDeviceLinearAlgebra

using CUDA
using CUDA.CUSPARSE
using Krylov
using LinearAlgebra
using SparseArrays

include("partition.jl")
include("vector.jl")
include("vector_linalg.jl")
include("vector_broadcast.jl")
include("ghost.jl")
include("matrix.jl")
include("mul.jl")
include("gather.jl")
include("krylov_compat.jl")
include("poisson.jl")

export PartitionSpec, compute_partition_ranges, device_for_index, device_id
export MultiDeviceVector, MultiDeviceSparseMatrixCSR
export consistent!
export gather, mdla_solve
export poisson_matrix_2d

end
