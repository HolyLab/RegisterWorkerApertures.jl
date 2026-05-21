using ImageMagick
using JLD, Test
using ImageCore, ImageAxes, ImageFiltering, TestImages
using StaticArrays, Interpolations
using RegisterCore, RegisterDeformation, RegisterMismatchCommon
using RegisterWorkerApertures, RegisterDriver
using AxisArrays: AxisArray
using Aqua
Aqua.test_all(RegisterWorkerApertures;
    unbound_args = (broken=true,),
    stale_deps = (ignore=[:CUDA, :RegisterMismatch, :RegisterMismatchCuda],),
    piracies = (treat_as_own=[RegisterWorkerShell.load_mm_package],))

if !(haskey(ENV, "CI")&&(ENV["CI"] == "true"))
    include("apertured_cuda.jl")
else
    include("apertured.jl")
    include("apertured1.jl")
end
