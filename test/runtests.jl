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
    persistent_tasks = (broken=true,),
    stale_deps = (ignore=[:CUDA, :RegisterMismatch, :RegisterMismatchCuda],),
    piracies = (treat_as_own=[RegisterWorkerShell.load_mm_package],))
using ExplicitImports
@test check_no_implicit_imports(RegisterWorkerApertures) === nothing
@test check_no_stale_explicit_imports(RegisterWorkerApertures) === nothing
@test check_all_explicit_imports_via_owners(RegisterWorkerApertures) === nothing

if !(haskey(ENV, "CI")&&(ENV["CI"] == "true"))
    include("apertured_cuda.jl")
else
    include("apertured.jl")
    include("apertured1.jl")
end
