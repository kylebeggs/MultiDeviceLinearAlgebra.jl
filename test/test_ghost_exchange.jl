@testset "Ghost Exchange GPU" begin
    @testset "scatter! explicit" begin
        for ndev in DEVICE_COUNTS
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
        for ndev in DEVICE_COUNTS
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
        for ndev in DEVICE_COUNTS
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
        for ndev in DEVICE_COUNTS
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

    # Pinned at 2 devices on purpose: this is *about* one pair with unequal
    # send/recv slab sizes, not about device count. The cross-socket variant of
    # the same case lives in test_cross_socket.jl.
    @testset "Asymmetric ghost counts" begin
        NGPUS < 2 && return

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
        for ndev in DEVICE_COUNTS
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
        for ndev in DEVICE_COUNTS
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

    # Regression guard for #24. `buf .= x[idx]` and `x[idx] .= op.(x[idx], buf)` each
    # materialize a device temporary per neighbor per call, because
    # `getindex(::CuVector, ::CuVector)` is evaluated eagerly rather than fused into the
    # broadcast. `_gather!` / `_scatter_apply!` replaced both with single launches, so the
    # steady-state device allocation of an exchange is now exactly zero.
    #
    # `CUDA.@allocated` reads `CUDA.alloc_stats`, bumped on every `pool_alloc` — cache hits
    # included, and from any device or task. That is what makes it usable across the
    # `@sync`/`@async` device loop.
    #
    # Note the counter is a *Julia process* global (`const alloc_stats = AllocStats()` in
    # CUDA.jl's `src/memory.jl`), not a device or driver one. Another tenant's job on the same
    # GPU cannot perturb it, so this testset is safe to run on a shared host; the only
    # requirement is that no other task in *this* process allocates concurrently.
    @testset "scatter!/reduce! allocate no device temporaries" begin
        for ndev in DEVICE_COUNTS
            ndev < 2 && continue
            @testset "$ndev devices" begin
                n = 10 * ndev
                spec = compute_partition_ranges(n, ndev)
                ggi = _neighbor_ghost_indices(spec)
                ghost = GhostExchange(ggi, spec, Float64)

                x = attach_ghost(MultiDeviceVector(collect(1.0:n), spec), ghost)

                # First call JITs both kernels and populates the pool; only the steady
                # state is being asserted on.
                scatter!(x)
                reduce!(x, +)

                @test CUDA.@allocated(scatter!(x)) == 0
                @test CUDA.@allocated(reduce!(x, +)) == 0
            end
        end
    end

    @testset "scatter!/reduce! without exchange throws" begin
        # Not about device count — any legal partition of 20 elements will do.
        spec = compute_partition_ranges(20, min(NGPUS, 20))
        x = MultiDeviceVector(randn(20), spec)
        @test_throws ArgumentError scatter!(x)
        @test_throws ArgumentError reduce!(x, +)
    end
end

@testset "Host-staged fallback" begin
    @testset "probe cache and construction flags" begin
        # Same-device copies never need P2P
        @test MultiDeviceLinearAlgebra._p2p_copy_ok(0, 0)

        if NGPUS >= 2
            spec = compute_partition_ranges(20, 2)
            src_dev = device_id(spec, 2)
            dst_dev = device_id(spec, 1)

            # Deterministic and cached: repeated calls agree. Do NOT assert `true` —
            # on hosts with broken P2P the correct answer is `false`.
            r1 = MultiDeviceLinearAlgebra._p2p_copy_ok(src_dev, dst_dev)
            r2 = MultiDeviceLinearAlgebra._p2p_copy_ok(src_dev, dst_dev)
            @test r1 isa Bool
            @test r1 == r2

            ghost = GhostExchange(_neighbor_ghost_indices(spec), spec, Float64)
            for d in 1:2, (k, nbr) in enumerate(ghost.neighbors[d])
                @test ghost.p2p_ok[d][k] == MultiDeviceLinearAlgebra._p2p_copy_ok(
                    device_id(spec, nbr), device_id(spec, d)
                )
                @test length(ghost.host_buffers[d][k]) == max(
                    length(ghost.send_local_indices[d][k]),
                    length(ghost.recv_ghost_offsets[d][k]),
                )
            end
        end
    end

    @testset "forced host path: scatter!" begin
        for ndev in DEVICE_COUNTS
            ndev < 2 && continue
            @testset "$ndev devices" begin
                n = 10 * ndev
                spec = compute_partition_ranges(n, ndev)
                ggi = _neighbor_ghost_indices(spec)

                ghost = GhostExchange(ggi, spec, Float64)
                for d in 1:ndev
                    fill!(ghost.p2p_ok[d], false)
                end

                x = MultiDeviceVector(collect(1.0:n), spec)
                scatter!(x, ghost, spec)

                for d in 1:ndev
                    local_x_host = Array(ghost.local_x[d])
                    n_owned = length(spec.ranges[d])

                    @test local_x_host[1:n_owned] ≈ collect(Float64.(spec.ranges[d]))
                    for (i, g) in enumerate(ggi[d])
                        @test local_x_host[n_owned + i] ≈ Float64(g)
                    end
                end
            end
        end
    end

    @testset "forced host path: reduce! with +" begin
        for ndev in DEVICE_COUNTS
            ndev < 2 && continue
            @testset "$ndev devices" begin
                n = 10 * ndev
                spec = compute_partition_ranges(n, ndev)
                ggi = _neighbor_ghost_indices(spec)

                ghost = GhostExchange(ggi, spec, Float64)
                for d in 1:ndev
                    fill!(ghost.p2p_ok[d], false)
                end

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
                    @test x_host[i] ≈ 1.0 + 10.0 * n_contributors
                end
            end
        end
    end

    # Pinned at 2 devices for the same reason as "Asymmetric ghost counts" above.
    @testset "forced host path: asymmetric ghost counts" begin
        NGPUS < 2 && return

        n = 20
        spec = compute_partition_ranges(n, 2)

        # Unequal send/recv slab sizes exercise the max-sized shared host buffer
        # through both 5-arg copyto! directions
        ggi = [
            [first(spec.ranges[2]), first(spec.ranges[2]) + 1, first(spec.ranges[2]) + 2],
            [last(spec.ranges[1])],
        ]
        ghost = GhostExchange(ggi, spec, Float64)
        for d in 1:2
            fill!(ghost.p2p_ok[d], false)
        end

        x = MultiDeviceVector(collect(1.0:n), spec)
        scatter!(x, ghost, spec)

        for d in 1:2
            local_x_host = Array(ghost.local_x[d])
            n_owned = length(spec.ranges[d])
            for (i, g) in enumerate(ggi[d])
                @test local_x_host[n_owned + i] ≈ Float64(g)
            end
        end

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

        boundary_idx = last(spec.ranges[1])
        @test result_host[boundary_idx] ≈ 1.0 + 5.0

        for g in ggi[1]
            @test result_host[g] ≈ 1.0 + 5.0
        end
    end

    @testset "copy_exchange carries fallback state" begin
        ndev = NGPUS
        ndev < 2 && return

        n = 10 * ndev
        spec = compute_partition_ranges(n, ndev)
        ggi = _neighbor_ghost_indices(spec)

        ghost = GhostExchange(ggi, spec, Float64)
        for d in 1:ndev
            fill!(ghost.p2p_ok[d], false)
        end

        derived = MultiDeviceLinearAlgebra.copy_exchange(ghost, spec)

        @test derived.p2p_ok == ghost.p2p_ok
        for d in 1:ndev
            @test derived.p2p_ok[d] !== ghost.p2p_ok[d]
            @test length.(derived.host_buffers[d]) == length.(ghost.host_buffers[d])
            for k in eachindex(ghost.host_buffers[d])
                @test derived.host_buffers[d][k] !== ghost.host_buffers[d][k]
            end
        end

        x = MultiDeviceVector(collect(1.0:n), spec)
        scatter!(x, derived, spec)

        for d in 1:ndev
            local_x_host = Array(derived.local_x[d])
            n_owned = length(spec.ranges[d])
            for (i, g) in enumerate(ggi[d])
                @test local_x_host[n_owned + i] ≈ Float64(g)
            end
        end
    end
end
