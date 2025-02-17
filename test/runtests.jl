using ImageMagick
using Distributed, SharedArrays, JLD, Test
using ImageCore, ImageAxes, ImageFiltering, TestImages
using StaticArrays, Interpolations
using RegisterCore, RegisterDeformation, RegisterMismatchCommon
using AxisArrays: AxisArray

aperturedprocs = addprocs(2)
@everywhere using RegisterWorkerApertures, RegisterDriver

if !(haskey(ENV,"CI")&&(ENV["CI"]=="true"))
    include("apertured_cuda.jl")
else
    include("apertured.jl")
    include("apertured1.jl")
end
