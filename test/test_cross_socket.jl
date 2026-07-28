# Cross-socket (NUMA-crossing) device-pair coverage.
#
# Every other GPU testset partitions over CUDA devices `0:ndev-1`, so even a
# 4-device run only ever touches devices sharing one NUMA node. On a multi-socket
# host the `SYS`-class pairs — the ones a mistranslating IOMMU corrupts hardest,
# and the ones `_p2p_copy_ok` and its host-staging fallback exist to protect —
# were never exercised. `FAR_PAIR` (test/runtests.jl) picks the most
# topologically distant pair the host offers.

@testset "Cross-socket device pair" begin
    if FAR_PAIR === nothing
        @info "Skipping cross-socket tests: fewer than 2 CUDA devices visible"
    else
        devs = collect(FAR_PAIR.devices)
        if !FAR_PAIR.cross_numa
            @info "FAR_PAIR $(FAR_PAIR.devices) does not cross a NUMA node " *
                "(nodes $(FAR_PAIR.nodes)); exercising the widest available pair instead"
        end

        n = 20
        pair_spec() = compute_partition_ranges(n; devices = devs)

        @testset "probe flags and buffer sizing" begin
            spec = pair_spec()
            @test collect(spec.devices) == devs

            ghost = GhostExchange(_neighbor_ghost_indices(spec), spec, Float64)

            for d in 1:2, (k, nbr) in enumerate(ghost.neighbors[d])
                # Deliberately not asserting `true`: on a host with broken P2P,
                # `false` is the correct answer and the fallback must carry it.
                @test ghost.p2p_ok[d][k] == MultiDeviceLinearAlgebra._p2p_copy_ok(
                    device_id(spec, nbr), device_id(spec, d)
                )
                @test length(ghost.host_buffers[d][k]) == max(
                    length(ghost.send_local_indices[d][k]),
                    length(ghost.recv_ghost_offsets[d][k]),
                )
            end

            # Both directions across the pair were actually probed
            cache = MultiDeviceLinearAlgebra._p2p_probe_cache
            @test haskey(cache, (devs[1], devs[2]))
            @test haskey(cache, (devs[2], devs[1]))
        end

        @testset "scatter! (P2P path)" begin
            spec = pair_spec()
            ggi = _neighbor_ghost_indices(spec)
            ghost = GhostExchange(ggi, spec, Float64)

            x = MultiDeviceVector(collect(1.0:n), spec)
            scatter!(x, ghost, spec)

            for d in 1:2
                local_x_host = Array(ghost.local_x[d])
                n_owned = length(spec.ranges[d])

                @test local_x_host[1:n_owned] ≈ collect(Float64.(spec.ranges[d]))
                for (i, g) in enumerate(ggi[d])
                    @test local_x_host[n_owned + i] ≈ Float64(g)
                end
            end
        end

        @testset "reduce! with + (P2P path)" begin
            spec = pair_spec()
            ggi = _neighbor_ghost_indices(spec)
            ghost = GhostExchange(ggi, spec, Float64)

            for d in 1:2
                CUDA.device!(device_id(spec, d))
                n_owned = length(spec.ranges[d])
                n_ghost = length(ggi[d])
                ghost.local_x[d] .= vcat(
                    CUDA.ones(Float64, n_owned),
                    CUDA.fill(10.0, n_ghost),
                )
            end

            x = MultiDeviceVector{Float64}(undef, spec)
            for d in 1:2
                CUDA.device!(device_id(spec, d))
                x.partitions[d] .= 0.0
            end

            reduce!(x, ghost, spec, +)
            x_host = Array(gather(x))

            for i in 1:n
                owner, _ = device_for_index(spec, i)
                n_contributors = count(d -> d != owner && i in ggi[d], 1:2)
                @test x_host[i] ≈ 1.0 + 10.0 * n_contributors
            end
        end

        @testset "asymmetric ghost counts (P2P path)" begin
            spec = pair_spec()

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

            @test result_host[last(spec.ranges[1])] ≈ 1.0 + 5.0
            for g in ggi[1]
                @test result_host[g] ≈ 1.0 + 5.0
            end
        end

        @testset "forced host staging: scatter! and reduce!" begin
            spec = pair_spec()
            ggi = _neighbor_ghost_indices(spec)
            ghost = GhostExchange(ggi, spec, Float64)
            for d in 1:2
                fill!(ghost.p2p_ok[d], false)
            end
            # Guard: if a refactor stops honouring these flags, this testset
            # would silently re-test the P2P path instead of the fallback.
            @test all(d -> all(!, ghost.p2p_ok[d]), 1:2)

            x = MultiDeviceVector(collect(1.0:n), spec)
            scatter!(x, ghost, spec)

            for d in 1:2
                local_x_host = Array(ghost.local_x[d])
                n_owned = length(spec.ranges[d])

                @test local_x_host[1:n_owned] ≈ collect(Float64.(spec.ranges[d]))
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
                    CUDA.fill(10.0, n_ghost),
                )
            end

            x_result = MultiDeviceVector{Float64}(undef, spec)
            for d in 1:2
                CUDA.device!(device_id(spec, d))
                x_result.partitions[d] .= 0.0
            end

            reduce!(x_result, ghost, spec, +)
            result_host = Array(gather(x_result))

            for i in 1:n
                owner, _ = device_for_index(spec, i)
                n_contributors = count(d -> d != owner && i in ggi[d], 1:2)
                @test result_host[i] ≈ 1.0 + 10.0 * n_contributors
            end
        end

        @testset "forced host staging: asymmetric ghost counts" begin
            spec = pair_spec()

            # Unequal send/recv slab sizes drive the shared max-sized host buffer
            # through both 5-arg copyto! directions, now across a SYS-class pair
            ggi = [
                [first(spec.ranges[2]), first(spec.ranges[2]) + 1, first(spec.ranges[2]) + 2],
                [last(spec.ranges[1])],
            ]
            ghost = GhostExchange(ggi, spec, Float64)
            for d in 1:2
                fill!(ghost.p2p_ok[d], false)
            end
            @test all(d -> all(!, ghost.p2p_ok[d]), 1:2)

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

            @test result_host[last(spec.ranges[1])] ≈ 1.0 + 5.0
            for g in ggi[1]
                @test result_host[g] ≈ 1.0 + 5.0
            end
        end

        @testset "SpMV round-trip" begin
            n_mat = 200
            spec = compute_partition_ranges(n_mat; devices = devs)

            A_cpu = sprand(Float64, n_mat, n_mat, 0.05) + 10.0 * sparse(I, n_mat, n_mat)
            x_cpu = randn(n_mat)
            y_expected = A_cpu * x_cpu

            A_md = MultiDeviceSparseMatrixCSR(A_cpu, spec)
            @test collect(A_md.row_spec.devices) == devs
            @test gather(A_md) ≈ A_cpu

            x_md = MultiDeviceVector(x_cpu, spec)
            y_md = MultiDeviceVector(zeros(n_mat), spec)

            mul!(y_md, A_md, x_md)
            @test gather(y_md) ≈ y_expected rtol = 1.0e-10

            # Same SpMV with the halo exchange forced through host staging
            for d in 1:2
                fill!(A_md.ghost_exchange.p2p_ok[d], false)
            end
            @test all(d -> all(!, A_md.ghost_exchange.p2p_ok[d]), 1:2)

            y_host_staged = MultiDeviceVector(zeros(n_mat), spec)
            mul!(y_host_staged, A_md, x_md)
            @test gather(y_host_staged) ≈ y_expected rtol = 1.0e-10
        end
    end
end
