"""
    poisson_matrix_2d(nx, ny; T = Float64) → SparseMatrixCSC

Standard 5-point finite-difference Laplacian on an `nx × ny` grid with Dirichlet boundary
conditions. Grid point `(i, j)` is unknown `(j - 1) * nx + i`.
"""
function poisson_matrix_2d(nx::Int, ny::Int; T::Type = Float64)
    hx = one(T) / (nx + 1)
    hy = one(T) / (ny + 1)

    # 1D Laplacian: [-1, 2, -1] / h^2
    function laplacian_1d(n, h)
        e = ones(T, n)
        return spdiagm(-1 => -e[1:(n - 1)], 0 => 2 * e, 1 => -e[1:(n - 1)]) / h^2
    end

    Tx = laplacian_1d(nx, hx)
    Ty = laplacian_1d(ny, hy)
    Ix = sparse(one(T) * I, nx, nx)
    Iy = sparse(one(T) * I, ny, ny)

    return kron(Iy, Tx) + kron(Ty, Ix)
end

"""
    prolongation_matrix_2d(nx, ny; bx = 2, by = 2, T = Float64) → SparseMatrixCSC

Piecewise-constant (injection) prolongation for the `nx × ny` grid that
[`poisson_matrix_2d`](@ref) discretizes: fine point `(i, j)` belongs to the aggregate covering
the `bx × by` block it falls in, with unit weight. The result is `nx*ny × nxc*nyc` where
`nxc = cld(nx, bx)` and `nyc = cld(ny, by)`, so blocks along the far edges may be partial.

Exactly one nonzero per row, which makes `P' * A * P` the aggregate-summed Galerkin operator —
the standard two-level coarse operator, and a fixture for the distributed triple product with
no algebraic-multigrid dependency.

Both indexings are column-major — fine point `(i, j)` is `(j - 1) * nx + i`, coarse aggregate
`(ia, ja)` is `(ja - 1) * nxc + ia`. That correspondence is what keeps the coarse numbering
monotone in the fine one, so a contiguous block of fine unknowns maps to a contiguous block of
coarse ones and the distributed halo stays a surface rather than becoming all-to-all.
"""
function prolongation_matrix_2d(nx::Int, ny::Int; bx::Int = 2, by::Int = 2, T::Type = Float64)
    @assert nx > 0 && ny > 0 "Grid dimensions must be positive, got ($nx, $ny)"
    @assert bx > 0 && by > 0 "Aggregate dimensions must be positive, got ($bx, $by)"

    nxc = cld(nx, bx)
    nyc = cld(ny, by)

    rows = Vector{Int}(undef, nx * ny)
    cols = Vector{Int}(undef, nx * ny)
    for j in 1:ny, i in 1:nx
        fine = (j - 1) * nx + i
        ia = cld(i, bx)
        ja = cld(j, by)
        rows[fine] = fine
        cols[fine] = (ja - 1) * nxc + ia
    end

    return sparse(rows, cols, ones(T, nx * ny), nx * ny, nxc * nyc)
end
