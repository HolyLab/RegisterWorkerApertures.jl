using ImageMagick
using JLD, Test
using ImageCore, ImageAxes, ImageFiltering, TestImages
using StaticArrays, Interpolations
using RegisterCore, RegisterDeformation, RegisterMismatchCommon
using RegisterWorkerApertures, RegisterDriver
using AxisArrays: AxisArray

if !(haskey(ENV,"CI")&&(ENV["CI"]=="true"))
    include("apertured_cuda.jl")
else
    include("apertured.jl")
    include("apertured1.jl")
end
