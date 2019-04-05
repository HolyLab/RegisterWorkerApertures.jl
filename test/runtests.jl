using ImageMagick
using Distributed, SharedArrays, JLD, Test
using Images, TestImages, StaticArrays, Interpolations
using RegisterCore, RegisterDeformation, RegisterMismatchCommon
using RegisterDriver, RegisterWorkerApertures

aperturedprocs = addprocs(2)
@sync for p in aperturedprocs
    @spawnat p eval(quote
        using Pkg
        Pkg.activate(".")
        Pkg.instantiate()
        using StaticArrays
        using RegisterWorkerApertures
    end)
end

include("apertured.jl")
include("apertured1.jl")
if !(haskey(ENV,"CI")&&(ENV["CI"]=="true"))
    include("apertured_cuda.jl")
end
