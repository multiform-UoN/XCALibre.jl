# =============================================================================
# Viscoelastic Stability Study Script
# =============================================================================
println("Running Viscoelastic Stability Study")

function run_case(script_name, args...)
    cmd = `julia --project=/Volumes/OpenFOAM/XCALibre.jl /Volumes/OpenFOAM/XCALibre.jl/examples/nonNewtonian/$script_name $args`
    println("--------------------------------------------------")
    println("Running: ", cmd)
    try
        # Run command and capture output to standard out
        run(cmd)
    catch e
        println("=> Execution failed!")
    end
end

println("\n=== 1. Baseline Stokes ===")
run_case("stokes_incomp_channel.jl")

println("\n=== 2. Maxwell (varying lambda) ===")
# args: lambda_p, dt, n_iter
for lam in [0.1, 1.0, 10.0]
    run_case("maxwell_incomp_channel.jl", string(lam), "0.01", "10")
end

println("\n=== 3. Oldroyd-B with advection (varying lambda) ===")
for lam in [0.1, 1.0, 10.0, 50.0, 100.0]
    run_case("oldroyd_incomp_channel.jl", string(lam), "0.01", "10")
end

println("\n=== 5. Oldroyd-B (timestep dependence, lambda=10) ===")
for dt in [0.001, 0.01, 0.1, 1.0]
    run_case("oldroyd_incomp_channel.jl", "10.0", string(dt), "10")
end

println("\n=== 4. Objective Rates (Jaumann vs Stretching-UCD) ===")
run_case("maxwell_objective_rates_channel.jl", "Jaumann")
run_case("maxwell_objective_rates_channel.jl", "StretchingOnlyUCD")
