using ImageMagick
using JLD, Test
using ImageCore, ImageAxes, ImageFiltering, TestImages
using StaticArrays, Interpolations
using RegisterCore, RegisterDeformation, RegisterMismatchCommon
using RegisterWorkerApertures, RegisterDriver, RegisterWorkerShell
using AxisArrays: AxisArray
using Aqua
Aqua.test_all(RegisterWorkerApertures;
    unbound_args = (broken=true,),
    persistent_tasks = (broken = VERSION < v"1.11",),
    stale_deps = (ignore=[:CUDA, :RegisterMismatch, :RegisterMismatchCuda],),
    piracies = (treat_as_own=[RegisterWorkerShell.load_mm_package],))
using ExplicitImports
@test check_no_implicit_imports(RegisterWorkerApertures) === nothing
@test check_no_stale_explicit_imports(RegisterWorkerApertures) === nothing
@test check_all_explicit_imports_via_owners(RegisterWorkerApertures) === nothing

@testset "default thresh tracks the normalization" begin
    fixed = rand(Float32, 16, 20)
    nodes = map(d -> range(1, stop = size(fixed, d), length = 5), (1, 2))
    scale = (0.5)^ndims(fixed) / prod(map(length, nodes))

    apix = Apertures(fixed, nodes, (4, 4), 1.0)
    @test apix.normalization == :pixels
    @test apix.thresh ≈ scale * length(fixed)

    # An aperture is rejected by comparing its mismatch to `thresh`, so the
    # threshold has to carry the units of the mismatch denominator.
    aint = Apertures(fixed, nodes, (4, 4), 1.0; normalization = :intensity)
    @test aint.normalization == :intensity
    @test aint.thresh ≈ scale * sum(abs2, fixed)
    @test aint.thresh != apix.thresh

    # An explicit thresh is taken as given under either normalization.
    @test Apertures(fixed, nodes, (4, 4), 1.0;
        normalization = :intensity, thresh = 0.25).thresh == 0.25

    @test_throws "normalization must be :pixels or :intensity" Apertures(
        fixed, nodes, (4, 4), 1.0; normalization = :intensty)
end

if !(haskey(ENV, "CI")&&(ENV["CI"] == "true"))
    include("apertured_cuda.jl")
else
    include("apertured.jl")
    include("apertured1.jl")
end
