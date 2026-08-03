# AirspeedVelocity entry point: `benchpkg` runs this file against *both* revisions of a PR,
# so every benchmark here must use API that exists on the PR's base branch too. Adding a
# benchmark for a brand-new function makes the base run fail — guard it with `isdefined`.
#
# CPU-only by necessity. GitHub-hosted runners have no GPU, so what a pull request can compare
# is the host-side construction path: matrix assembly, the CSC→CSR transpose, ghost discovery
# and topology, and local column remapping. All of that is O(nnz) and all of it runs before a
# single byte reaches a device. The multi-GPU timings — SpMV, halo exchange, Krylov solve, and
# the device-scaling speedups — live in `benchmark/gpu.jl`, which is run by hand on GPU
# hardware. A green benchmark comment on a PR therefore says nothing about device performance.
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using MultiDeviceLinearAlgebra

const MDLA = MultiDeviceLinearAlgebra

const SUITE = BenchmarkGroup()

# 2D Poisson 5-point stencil — the sparsity pattern the package is built for, and the one
# `scripts/check_poisson.jl` and `benchmark/gpu.jl` also use.
const NX = 250
const A = poisson_matrix_2d(NX, NX)
const N = size(A, 1)

# The CSC→CSR conversion `MultiDeviceSparseMatrixCSR` performs before partitioning
# (`src/matrix.jl`). It is a full sparse transpose and one of the larger host costs.
_to_csr(M) = SparseMatrixCSC(sparse(M'))

const AT = _to_csr(A)
const ROWPTR = AT.colptr
const COLVAL = AT.rowval

const DEVICE_COUNTS = (2, 4, 8)

SUITE["assembly"]["poisson_matrix_2d $(NX)²"] = @benchmarkable poisson_matrix_2d($NX, $NX)
SUITE["assembly"]["CSC→CSR transpose"] = @benchmarkable _to_csr($A)

# Deliberately not benchmarked: `compute_partition_ranges`. It is O(ndevices) — eight loop
# iterations at most — and the compiler eliminates the call outright, so it clocks at 0.001 ns
# and its ratio column is pure noise.
for ndev in DEVICE_COUNTS
    spec = compute_partition_ranges(N, ndev)
    ggi, = MDLA._compute_ghost_map(ROWPTR, COLVAL, spec)

    # One device's row block of column indices, sliced the way `src/matrix.jl` slices it
    # before remapping to local numbering and uploading.
    r = spec.ranges[1]
    colval_block = COLVAL[ROWPTR[first(r)]:(ROWPTR[last(r) + 1] - 1)]

    SUITE["ghost"]["_compute_ghost_map, $ndev dev"] =
        @benchmarkable MDLA._compute_ghost_map($ROWPTR, $COLVAL, $spec)
    SUITE["ghost"]["_compute_ghost_topology, $ndev dev"] =
        @benchmarkable MDLA._compute_ghost_topology($ggi, $spec)
    SUITE["ghost"]["_remap_colval, $ndev dev"] =
        @benchmarkable MDLA._remap_colval($colval_block, $r, $(ggi[1]))
end

# Ghost discovery for a *rectangular* operator, where column ownership is resolved against a
# separate column partition. `hasmethod` rather than `isdefined`: the function has existed all
# along, only the four-argument method is new, and this same file runs against the PR's base
# revision where that method does not exist.
if hasmethod(
        MDLA._compute_ghost_map,
        Tuple{Vector{Int}, Vector{Int}, MDLA.PartitionSpec, MDLA.PartitionSpec},
    )
    # Injection prolongation over the same grid: NX² fine rows, one nonzero each, coarsened 2×2.
    P = prolongation_matrix_2d(NX, NX; bx = 2, by = 2)
    PT = _to_csr(P)
    P_ROWPTR = PT.colptr
    P_COLVAL = PT.rowval
    NC = size(P, 2)

    SUITE["assembly"]["prolongation_matrix_2d $(NX)²"] =
        @benchmarkable prolongation_matrix_2d($NX, $NX; bx = 2, by = 2)

    for ndev in DEVICE_COUNTS
        row_spec = compute_partition_ranges(N, ndev)
        col_spec = compute_partition_ranges(NC, ndev)
        SUITE["ghost"]["_compute_ghost_map rectangular, $ndev dev"] =
            @benchmarkable MDLA._compute_ghost_map(
            $P_ROWPTR, $P_COLVAL, $row_spec, $col_spec
        )
    end
end
