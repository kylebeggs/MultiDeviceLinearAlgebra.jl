@testset "Ghost Exchange GPU" begin
    @testset "scatter! explicit" begin
        for ndev in 1:min(NGPUS, 4)
            ndev < 2 && continue
            @testset "$ndev devices" begin
                n = 10 * ndev
                spec = compute_partition_ranges(n, ndev)

                # Each device ghosts its immediate neighbor's boundary index
                ggi = Vector{Vector{Int}}(undef, ndev)
                for d in 1:ndev
                    ghosts = Int[]
                    if d > 1
                        push!(ghosts, last(spec.ranges[d - 1]))
                    end
                    if d < ndev
                        push!(ghosts, first(spec.ranges[d + 1]))
                    end
                    ggi[d] = sort!(ghosts)
                end

                ghost = GhostExchange(ggi, spec, Float64)
                x = MultiDeviceVector(collect(1.0:n), spec)

                scatter!(x, ghost, spec)

                for d in 1:ndev
                    local_x_host = Array(ghost.local_x[d])
                    n_owned = length(spec.ranges[d])
                    n_ghost = length(ggi[d])

                    # Owned portion matches x
                    @test local_x_host[1:n_owned] ≈ collect(Float64.(spec.ranges[d]))

                    # Ghost portion has correct values
                    for (i, g) in enumerate(ggi[d])
                        @test local_x_host[n_owned + i] ≈ Float64(g)
                    end
                end
            end
        end
    end

    @testset "reduce! with +" begin
        for ndev in 1:min(NGPUS, 4)
            ndev < 2 && continue
            @testset "$ndev devices" begin
                n = 10 * ndev
                spec = compute_partition_ranges(n, ndev)

                ggi = Vector{Vector{Int}}(undef, ndev)
                for d in 1:ndev
                    ghosts = Int[]
                    if d > 1
                        push!(ghosts, last(spec.ranges[d - 1]))
                    end
                    if d < ndev
                        push!(ghosts, first(spec.ranges[d + 1]))
                    end
                    ggi[d] = sort!(ghosts)
                end

                ghost = GhostExchange(ggi, spec, Float64)

                # Fill local_x: owned = 1.0, ghost = 10.0
                for d in 1:ndev
                    CUDA.device!(device_id(spec, d))
                    n_owned = length(spec.ranges[d])
                    n_ghost = length(ggi[d])
                    ghost.local_x[d] .= vcat(
                        CUDA.ones(Float64, n_owned),
                        CUDA.fill(10.0, n_ghost),
                    )
                end

                x = MultiDeviceVector{Float64}(undef, spec)
                for d in 1:ndev
                    CUDA.device!(device_id(spec, d))
                    x.partitions[d] .= 0.0
                end

                reduce!(x, ghost, spec, +)

                x_host = Array(gather(x))

                for i in 1:n
                    owner, _ = device_for_index(spec, i)
                    n_contributors = count(
                        d -> d != owner && i in ggi[d], 1:ndev
                    )
                    # owned value = 1.0, each ghost contributor adds 10.0
                    @test x_host[i] ≈ 1.0 + 10.0 * n_contributors
                end
            end
        end
    end

    @testset "reduce! with max" begin
        for ndev in 1:min(NGPUS, 4)
            ndev < 2 && continue
            @testset "$ndev devices" begin
                n = 10 * ndev
                spec = compute_partition_ranges(n, ndev)

                ggi = Vector{Vector{Int}}(undef, ndev)
                for d in 1:ndev
                    ghosts = Int[]
                    if d > 1
                        push!(ghosts, last(spec.ranges[d - 1]))
                    end
                    if d < ndev
                        push!(ghosts, first(spec.ranges[d + 1]))
                    end
                    ggi[d] = sort!(ghosts)
                end

                ghost = GhostExchange(ggi, spec, Float64)

                # Fill local_x: owned = device_id, ghost = 100.0
                for d in 1:ndev
                    CUDA.device!(device_id(spec, d))
                    n_owned = length(spec.ranges[d])
                    n_ghost = length(ggi[d])
                    ghost.local_x[d] .= vcat(
                        CUDA.fill(Float64(d), n_owned),
                        CUDA.fill(100.0, n_ghost),
                    )
                end

                x = MultiDeviceVector{Float64}(undef, spec)
                for d in 1:ndev
                    CUDA.device!(device_id(spec, d))
                    x.partitions[d] .= 0.0
                end

                reduce!(x, ghost, spec, max)

                x_host = Array(gather(x))

                for i in 1:n
                    owner, _ = device_for_index(spec, i)
                    has_ghost_contrib = any(
                        d -> d != owner && i in ggi[d], 1:ndev
                    )
                    if has_ghost_contrib
                        @test x_host[i] ≈ 100.0
                    else
                        @test x_host[i] ≈ Float64(owner)
                    end
                end
            end
        end
    end

    @testset "scatter! then reduce! round-trip" begin
        for ndev in 1:min(NGPUS, 4)
            ndev < 2 && continue
            @testset "$ndev devices" begin
                n = 10 * ndev
                spec = compute_partition_ranges(n, ndev)

                ggi = Vector{Vector{Int}}(undef, ndev)
                for d in 1:ndev
                    ghosts = Int[]
                    if d > 1
                        push!(ghosts, last(spec.ranges[d - 1]))
                    end
                    if d < ndev
                        push!(ghosts, first(spec.ranges[d + 1]))
                    end
                    ggi[d] = sort!(ghosts)
                end

                ghost = GhostExchange(ggi, spec, Float64)
                x_orig = MultiDeviceVector(collect(1.0:n), spec)

                # scatter! populates local_x with owned + ghost values
                scatter!(x_orig, ghost, spec)

                # reduce! with + should give: owned + sum of ghost copies pointing here
                x_result = MultiDeviceVector{Float64}(undef, spec)
                for d in 1:ndev
                    CUDA.device!(device_id(spec, d))
                    x_result.partitions[d] .= 0.0
                end

                reduce!(x_result, ghost, spec, +)
                result_host = Array(gather(x_result))

                for i in 1:n
                    owner, _ = device_for_index(spec, i)
                    n_ghost_copies = count(
                        d -> d != owner && i in ggi[d], 1:ndev
                    )
                    @test result_host[i] ≈ Float64(i) * (1 + n_ghost_copies)
                end
            end
        end
    end

    @testset "Asymmetric ghost counts" begin
        ndev = min(NGPUS, 2)
        ndev < 2 && return

        n = 20
        spec = compute_partition_ranges(n, 2)

        # Device 1 needs 3 ghosts from device 2, device 2 needs 1 from device 1
        ggi = [
            [first(spec.ranges[2]), first(spec.ranges[2]) + 1, first(spec.ranges[2]) + 2],
            [last(spec.ranges[1])],
        ]
        ghost = GhostExchange(ggi, spec, Float64)

        x = MultiDeviceVector(collect(1.0:n), spec)
        scatter!(x, ghost, spec)

        for d in 1:2
            local_x_host = Array(ghost.local_x[d])
            n_owned = length(spec.ranges[d])
            for (i, g) in enumerate(ggi[d])
                @test local_x_host[n_owned + i] ≈ Float64(g)
            end
        end

        # Test reduce with asymmetric ghosts
        for d in 1:2
            CUDA.device!(device_id(spec, d))
            n_owned = length(spec.ranges[d])
            n_ghost = length(ggi[d])
            ghost.local_x[d] .= vcat(
                CUDA.ones(Float64, n_owned),
                CUDA.fill(5.0, n_ghost),
            )
        end

        x_result = MultiDeviceVector{Float64}(undef, spec)
        for d in 1:2
            CUDA.device!(device_id(spec, d))
            x_result.partitions[d] .= 0.0
        end

        reduce!(x_result, ghost, spec, +)
        result_host = Array(gather(x_result))

        # Boundary indices that are ghosted get contributions
        boundary_idx = last(spec.ranges[1])
        @test result_host[boundary_idx] ≈ 1.0 + 5.0  # owned + 1 ghost contrib

        for g in ggi[1]
            @test result_host[g] ≈ 1.0 + 5.0  # owned + 1 ghost contrib
        end
    end
end

@testset "Convenience scatter!/reduce! (vector-owned exchange)" begin
    @testset "scatter!(x) convenience" begin
        for ndev in 1:min(NGPUS, 4)
            ndev < 2 && continue
            @testset "$ndev devices" begin
                n = 10 * ndev
                spec = compute_partition_ranges(n, ndev)
                ggi = _neighbor_ghost_indices(spec)
                ghost = GhostExchange(ggi, spec, Float64)

                x = attach_ghost(MultiDeviceVector(collect(1.0:n), spec), ghost)
                scatter!(x)

                for d in 1:ndev
                    local_x_host = Array(ghost.local_x[d])
                    n_owned = length(spec.ranges[d])
                    for (i, g) in enumerate(ggi[d])
                        @test local_x_host[n_owned + i] ≈ Float64(g)
                    end
                end
            end
        end
    end

    @testset "reduce!(x, op) convenience" begin
        for ndev in 1:min(NGPUS, 4)
            ndev < 2 && continue
            @testset "$ndev devices" begin
                n = 10 * ndev
                spec = compute_partition_ranges(n, ndev)
                ggi = _neighbor_ghost_indices(spec)
                ghost = GhostExchange(ggi, spec, Float64)

                for d in 1:ndev
                    CUDA.device!(device_id(spec, d))
                    n_owned = length(spec.ranges[d])
                    n_ghost = length(ggi[d])
                    ghost.local_x[d] .= vcat(
                        CUDA.ones(Float64, n_owned),
                        CUDA.fill(10.0, n_ghost),
                    )
                end

                x = attach_ghost(MultiDeviceVector{Float64}(undef, spec), ghost)
                for d in 1:ndev
                    CUDA.device!(device_id(spec, d))
                    x.partitions[d] .= 0.0
                end

                reduce!(x, +)

                x_host = Array(gather(x))
                for i in 1:n
                    owner, _ = device_for_index(spec, i)
                    n_contributors = count(
                        d -> d != owner && i in ggi[d], 1:ndev
                    )
                    @test x_host[i] ≈ 1.0 + 10.0 * n_contributors
                end
            end
        end
    end

    @testset "scatter!/reduce! without exchange throws" begin
        spec = compute_partition_ranges(20, min(NGPUS, 2))
        x = MultiDeviceVector(randn(20), spec)
        @test_throws ArgumentError scatter!(x)
        @test_throws ArgumentError reduce!(x, +)
    end
end
