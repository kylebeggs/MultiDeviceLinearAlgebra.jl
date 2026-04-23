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
