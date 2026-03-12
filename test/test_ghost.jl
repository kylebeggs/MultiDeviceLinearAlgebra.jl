using MultiDeviceLinearAlgebra: _compute_ghost_map, _remap_colval

function _to_csr(A::SparseArrays.SparseMatrixCSC)
    At = SparseMatrixCSC(sparse(A'))
    return At.colptr, At.rowval
end

@testset "Ghost Exchange" begin
    @testset "_compute_ghost_map" begin
        @testset "Tridiagonal, 2 devices" begin
            n = 8
            A = spdiagm(-1 => ones(n - 1), 0 => 2 * ones(n), 1 => ones(n - 1))
            rowptr, colval = _to_csr(A)
            spec = compute_partition_ranges(n, 2)

            ggi, nbrs, sli, rgo = _compute_ghost_map(rowptr, colval, spec)

            # Device 1 (rows 1:4) needs col 5 from device 2
            @test ggi[1] == [5]
            # Device 2 (rows 5:8) needs col 4 from device 1
            @test ggi[2] == [4]

            @test nbrs[1] == [2]
            @test nbrs[2] == [1]

            @test rgo[1] == [1:1]
            @test rgo[2] == [1:1]

            # Device 1 sends local index 4 (global 4) to device 2
            @test sli[1] == [[4]]
            # Device 2 sends local index 1 (global 5) to device 1
            @test sli[2] == [[1]]
        end

        @testset "Tridiagonal, 4 devices" begin
            n = 12
            A = spdiagm(-1 => ones(n - 1), 0 => 2 * ones(n), 1 => ones(n - 1))
            rowptr, colval = _to_csr(A)
            spec = compute_partition_ranges(n, 4)

            ggi, nbrs, sli, rgo = _compute_ghost_map(rowptr, colval, spec)

            # Boundary devices have 1 neighbor, interior devices have 2
            @test length(nbrs[1]) == 1
            @test length(nbrs[2]) == 2
            @test length(nbrs[3]) == 2
            @test length(nbrs[4]) == 1

            # Each boundary device needs 1 ghost, each interior needs 2
            @test length(ggi[1]) == 1
            @test length(ggi[2]) == 2
            @test length(ggi[3]) == 2
            @test length(ggi[4]) == 1

            # Ghost indices are in neighboring device ranges
            @test all(g -> g in spec.ranges[2], ggi[1])
            @test all(g -> g in spec.ranges[1] || g in spec.ranges[3], ggi[2])
            @test all(g -> g in spec.ranges[2] || g in spec.ranges[4], ggi[3])
            @test all(g -> g in spec.ranges[3], ggi[4])
        end

        @testset "Diagonal matrix (no ghosts)" begin
            n = 8
            A = spdiagm(0 => ones(n))
            rowptr, colval = _to_csr(A)
            spec = compute_partition_ranges(n, 2)

            ggi, nbrs, sli, rgo = _compute_ghost_map(rowptr, colval, spec)

            @test ggi[1] == Int[]
            @test ggi[2] == Int[]
            @test nbrs[1] == Int[]
            @test nbrs[2] == Int[]
        end

        @testset "Single device (no ghosts)" begin
            n = 8
            A = sprand(Float64, n, n, 0.5) + 5.0 * sparse(I, n, n)
            rowptr, colval = _to_csr(A)
            spec = compute_partition_ranges(n, 1)

            ggi, nbrs, sli, rgo = _compute_ghost_map(rowptr, colval, spec)

            @test ggi[1] == Int[]
            @test nbrs[1] == Int[]
        end

        @testset "5-point stencil (Poisson)" begin
            nx = 4
            n = nx * nx
            A = poisson_matrix_2d(nx, nx)
            rowptr, colval = _to_csr(A)
            spec = compute_partition_ranges(n, 2)

            ggi, nbrs, sli, rgo = _compute_ghost_map(rowptr, colval, spec)

            @test !isempty(ggi[1])
            @test !isempty(ggi[2])
            @test nbrs[1] == [2]
            @test nbrs[2] == [1]

            @test all(g -> g in spec.ranges[2], ggi[1])
            @test all(g -> g in spec.ranges[1], ggi[2])

            # Total ghost offsets must cover all ghosts
            @test sum(length, rgo[1]) == length(ggi[1])
            @test sum(length, rgo[2]) == length(ggi[2])
        end

        @testset "Symmetric send/receive" begin
            n = 20
            A = sprand(Float64, n, n, 0.3) + 5.0 * sparse(I, n, n)
            A = A + A'  # symmetric sparsity
            rowptr, colval = _to_csr(A)
            spec = compute_partition_ranges(n, 4)

            ggi, nbrs, sli, rgo = _compute_ghost_map(rowptr, colval, spec)

            # For symmetric sparsity, if d1 is a neighbor of d2, d2 is a neighbor of d1
            for d in 1:4
                for nbr in nbrs[d]
                    @test d in nbrs[nbr]
                end
            end
        end
    end

    @testset "_remap_colval" begin
        @testset "Mixed owned and ghost" begin
            owned_range = 5:8
            ghost_global = [1, 2, 10]
            colval = Int32[5, 1, 6, 10, 7, 2, 8]

            remapped = _remap_colval(colval, owned_range, ghost_global)

            # owned: 5→1, 6→2, 7→3, 8→4
            # ghost: 1→5, 2→6, 10→7
            @test remapped == Int32[1, 5, 2, 7, 3, 6, 4]
        end

        @testset "No ghosts" begin
            owned_range = 1:4
            ghost_global = Int[]
            colval = Int32[1, 2, 3, 4, 2, 3]

            remapped = _remap_colval(colval, owned_range, ghost_global)
            @test remapped == colval
        end

        @testset "All ghosts" begin
            owned_range = 3:4
            ghost_global = [1, 2, 5, 6]
            colval = Int32[1, 5, 2, 6]

            remapped = _remap_colval(colval, owned_range, ghost_global)
            # n_owned=2, ghost: 1→3, 2→4, 5→5, 6→6
            @test remapped == Int32[3, 5, 4, 6]
        end

        @testset "Preserves element type" begin
            owned_range = 1:3
            ghost_global = [4, 5]
            colval = Int64[1, 4, 2, 5, 3]

            remapped = _remap_colval(colval, owned_range, ghost_global)
            @test eltype(remapped) == Int64
            @test remapped == Int64[1, 4, 2, 5, 3]
        end

        @testset "Round-trip consistency with ghost map" begin
            n = 12
            A = spdiagm(-1 => ones(n - 1), 0 => 2 * ones(n), 1 => ones(n - 1))
            rowptr, colval = _to_csr(A)
            spec = compute_partition_ranges(n, 3)

            ggi, _, _, _ = _compute_ghost_map(rowptr, colval, spec)

            for d in 1:3
                r = spec.ranges[d]
                rp_start = rowptr[first(r)]
                rp_end = rowptr[last(r) + 1] - 1
                local_colval = colval[rp_start:rp_end]

                remapped = _remap_colval(local_colval, r, ggi[d])

                n_owned = length(r)
                n_ghost = length(ggi[d])

                # All remapped values must be in [1, n_owned + n_ghost]
                @test all(1 .<= remapped .<= n_owned + n_ghost)
                # Owned columns remap to [1, n_owned]
                for (i, col) in enumerate(local_colval)
                    if col in r
                        @test remapped[i] <= n_owned
                    else
                        @test remapped[i] > n_owned
                    end
                end
            end
        end
    end
end
