include("shared.jl")

using Krylov  # loads MathOptScalingKrylovExt
using MathOptScaling: CurtisReidScaling, CurtisReidWorkspace, curtis_reid_scaling, curtis_reid_scaling!

@testset "README example" begin
    A = [
        100.0 0.0 3.0;
        2.0 4.0 0.0;
        -1.5 5.0 0.7;
    ]
    S, D = ruiz_equilibration(A)
    @test S ≈ Diagonal(D.row) * A * Diagonal(D.col)
    @test all(≈(1), maximum(abs, S; dims=1)) && all(≈(1), maximum(abs, S; dims=2))
end

run_backend_tests("CPU", identity)

@testset "CurtisReid" begin
    @testset "M$i / $fname" for (i, M) in enumerate(MATRICES), (fname, mk) in (("dense", identity), ("COO", x -> coo_arr(identity, x)))
        A = mk(Float64.(M))
        S, D = curtis_reid_scaling!(A)
        @test D isa CurtisReidScaling
        @test reconstructs(S, Float64.(M), D)
    end
end

@testset "CPU edge cases" begin
    @testset "zero matrix" begin
        for Z in (zeros(3, 4), coo_arr(identity, spzeros(3, 4)))
            S, D = ruiz_equilibration(Z)
            @test densify(S) == zeros(3, 4)
            @test all(isone, D.row) && all(isone, D.col)
        end
    end

    @testset "errors" begin
        A = MATRICES[1]
        for p in (0, Val(0), -1, NaN)
            @test_throws ArgumentError diagonal_norm_scaling(copy(A); p)
        end
        @test_throws ScalingConvergenceError ruiz_equilibration(copy(A); max_iter = 0, strict = true)
    end

    @testset "huge values stay finite" begin
        Huge = [1e200 0.0; 0.0 2e200]
        for B in (copy(Huge), coo_arr(identity, Huge))
            S, D = diagonal_norm_scaling(B; p = 2)
            @test all(isfinite, densify(S))
            @test all(!iszero, D.row) && all(!iszero, D.col)
        end
        for alpha in (0, 2)
            S, D = chambolle_pock_scaling(copy(Huge); alpha)
            @test all(isfinite, D.row) && all(isfinite, D.col)
            @test minimum(abs, diag(densify(S))) > 0
        end
    end

    @testset "Tomlin already converged" begin
        S, _ = tomlin_scaling([2.0 0.0; 0.0 8.0]; max_iter = 1, check = false)
        @test S ≈ I(2)
        @test tomlin_converged(TomlinWorkspace(S); eps = 1e-12)
    end
end
