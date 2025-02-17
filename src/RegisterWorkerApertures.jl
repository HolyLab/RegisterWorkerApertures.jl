module RegisterWorkerApertures

using ImageCore, CoordinateTransformations, Interpolations, SharedArrays
using RegisterCore, RegisterDeformation, RegisterFit, RegisterPenalty
using RegisterMismatchCommon, RegisterOptimize
# Note: RegisterMismatch/RegisterMismatchCuda is selected below
using RegisterWorkerShell  #, RegisterDriver

import RegisterWorkerShell: worker, init!, close!, load_mm_package

export Apertures, monitor, monitor!, worker, workerpid

mutable struct Apertures{A<:AbstractArray,T,K,N} <: AbstractWorker
    fixed::A
    nodes::NTuple{N,K}
    maxshift::NTuple{N,Int}
    affinepenalty::AffinePenalty{T,N}
    overlap::NTuple{N,Int}
    λrange::Union{T,Tuple{T,T}}
    thresh::T
    preprocess  # likely of type PreprocessSNF, but could be a function
    normalization::Symbol
    correctbias::Bool
    workerpid::Int
    dev::Int
    cuda_objects::Dict{Symbol,Any}
end

function load_mm_package(dev)
    if dev >= 0
        eval(:(using CUDA, RegisterMismatchCuda))
    else
        eval(:(using RegisterMismatch))
    end
    nothing
end

function init!(algorithm::Apertures)
    if algorithm.dev >= 0
        cuda_init!(algorithm)
    end
    nothing
end

function cuda_init!(algorithm)
    dev = CuDevice(algorithm.dev)
    global old_active_context
    try
        old_active_context = current_context()
        if old_active_context == nothing || device(old_active_context) != dev
            device!(dev)
        end
    catch e
        old_active_context = nothing
        device!(dev)
    end
    fixed = algorithm.fixed
    T = cudatype(eltype(fixed))
    d_fixed  = CuArray{T}(sdata(fixed))
    algorithm.cuda_objects[:d_fixed] = d_fixed
    algorithm.cuda_objects[:d_moving] = similar(d_fixed)
    gridsize = map(length, algorithm.nodes)
    aperture_width = default_aperture_width(algorithm.fixed, gridsize)
    algorithm.cuda_objects[:cms] = CMStorage{T}(undef, aperture_width, algorithm.maxshift)
end

function close!(algorithm::Apertures)
    if algorithm.dev >= 0
        if old_active_context != nothing
            activate(old_active_context)
        end
    end
    nothing
end

"""
`alg = Apertures(fixed, nodes, maxshift, λ, [preprocess=identity]; kwargs...)`
creates a worker-object for performing "apertured" (blocked)
registration.  `fixed` is the reference image, `nodes` specifies the
grid of apertures, `maxshift` represents the largest shift (in pixels)
that will be evaluated, and `λ` is the coefficient for the deformation
penalty (higher values enforce a more affine-like
deformation). `preprocess` allows you to apply a transformation (e.g.,
filtering) to the `moving` images before registration; `fixed` should
already have any such transformations applied.

Alternatively, `λ` may be specified as a `(λmin, λmax)` tuple, in
which case the "best" `λ` is chosen for you automatically via the
algorithm described in `auto_λ`.  If you `monitor` the variable
`datapenalty`, you can inspect the quality of the sigmoid used to
choose `λ`.

Registration is performed by calling `driver`.

## Example

Suppose your images are somewhat noisy, in which case a bit of
smoothing might help considerably.  Here we'll illustrate the use of a
pre-processing function, but see also `PreprocessSNF`.

```
   # Raw images are fixed0 and moving0, both two-dimensional
   pp = img -> imfilter_gaussian(img, [3, 3])
   fixed = pp(fixed0)
   # We'll use a 5x7 grid of apertures
   nodes = (linspace(1, size(fixed,1), 5), linspace(1, size(fixed,2), 7))
   # Allow shifts of up to 30 pixels in any direction
   maxshift = (30,30)
   # Try a range of λ values
   λrange = (1e-6, 100)

   # Create the algorithm-object
   alg = Apertures(fixed, nodes, maxshift, λrange, pp)

   # Monitor the datapenalty, the chosen value of λ, the deformation
   # u, and also collect the corrected (warped) image. By asking for
   # :warped0, we apply the warping to the unfiltered moving image
   # (:warped would refer to the filtered moving image).
   # We pre-allocate space for :warped0 to illustrate a trick for
   # reducing the overhead of communication between worker and driver
   # processes, even though this example uses just a single process
   # (see `monitor` for further detail).  The other arrays are small,
   # so we don't worry about overhead for them.
   mon = monitor(alg, (), Dict(:λs=>0, :datapenalty=>0, :λ=>0, :u=>0, :warped0 => Array(Float64, size(fixed))))

   # Run the algorithm
   mon = driver(algorithm, moving0, mon)

   # Plot the datapenalty and see how sigmoidal it is. Assumes you're
   # `using Immerse`.
   λs = mon[:λs]
   datapenalty = mon[:datapenalty]
   plot(x=λs, y=datapenalty, xintercept=[mon[:λ]], Geom.point, Geom.vline, Guide.xlabel("λ"), Guide.ylabel("Data penalty"), Scale.x_log10)
```

"""
function Apertures(fixed, nodes::NTuple{N,K}, maxshift, λrange, preprocess=identity; overlap=zeros(Int, N), normalization=:pixels, thresh_fac=(0.5)^ndims(fixed), thresh=nothing, correctbias::Bool=true, pid=1, dev=-1) where {K,N}
    gridsize = map(length, nodes)
    overlap_t = (overlap...,) #Make tuple
    length(overlap) == N || throw(DimensionMismatch("overlap must have $N entries"))
    nimages(fixed) == 1 || error("Register to a single image")
    isa(λrange, Number) || isa(λrange, Tuple{Number,Number}) || error("λrange must be a number or 2-tuple")
    if thresh == nothing
        thresh = (thresh_fac/prod(gridsize)) * (normalization==:pixels ? length(fixed) : sumabs2(fixed))
    end
    # T = eltype(fixed) <: AbstractFloat ? eltype(fixed) : Float32
    T = Float64   # Ipopt requires Float64
    λrange = isa(λrange, Number) ? T(λrange) : (T(first(λrange)), T(last(λrange)))
    Apertures{typeof(fixed),T,K,N}(fixed, nodes, maxshift, AffinePenalty{T,N}(nodes, first(λrange)), overlap_t, λrange, T(thresh), preprocess, normalization, correctbias, pid, dev, Dict{Symbol,Any}())
end

function worker(algorithm::Apertures, img, tindex, mon)
    moving0 = getindex_t(img, tindex)
    moving = algorithm.preprocess(moving0)
    gridsize = map(length, algorithm.nodes)
    use_cuda = algorithm.dev >= 0
    if use_cuda
        device!(CuDevice(algorithm.dev))
        d_fixed  = algorithm.cuda_objects[:d_fixed]
        d_moving = algorithm.cuda_objects[:d_moving]
        cms      = algorithm.cuda_objects[:cms]
        copyto!(d_moving, moving)
        cs = coords_spatial(img)
        aperture_centers = aperture_grid(map(d->size(img,d),cs), gridsize)
        mms = allocate_mmarrays(eltype(cms), gridsize, algorithm.maxshift)
        mismatch_apertures!(mms, d_fixed, d_moving, aperture_centers, cms; normalization=algorithm.normalization)
    else
        #mms = mismatch_apertures(algorithm.fixed, moving, gridsize, algorithm.maxshift; normalization=algorithm.normalization)
        cs = coords_spatial(img) #
        aperture_centers = aperture_grid(map(d->size(img,d),cs), gridsize)
        aperture_width = default_aperture_width(algorithm.fixed, gridsize, algorithm.overlap)  #
        mms = mismatch_apertures(algorithm.fixed, moving, aperture_centers, aperture_width, algorithm.maxshift; normalization=algorithm.normalization)  #
    end
    # displaymismatch(mms, thresh=10)
    if algorithm.correctbias
        correctbias!(mms)
    end
    E0 = zeros(size(mms))
    cs = Array{Any}(undef, size(mms))
    Qs = Array{Any}(undef, size(mms))
    thresh = algorithm.thresh
    for i = 1:length(mms)
        E0[i], cs[i], Qs[i] = qfit(mms[i], thresh; opt=false)
    end
    mmis = interpolate_mm!(mms)
    λrange = algorithm.λrange
    if isa(λrange, Number)
        ϕ, mismatch = RegisterOptimize.fixed_λ(cs, Qs, algorithm.nodes, algorithm.affinepenalty, mmis)
    else
        ϕ, mismatch, λ, λs, dp, quality = RegisterOptimize.auto_λ(cs, Qs, algorithm.nodes, algorithm.affinepenalty, mmis, λrange)
        monitor!(mon, :λ, λ)
        monitor!(mon, :λs, λs)
        monitor!(mon, :datapenalty, dp)
        monitor!(mon, :sigmoid_quality, quality)
    end
    monitor!(mon, :mismatch, mismatch)
    monitor!(mon, :u, ϕ.u)
    if haskey(mon, :warped)
        warped = warp(moving, ϕ)
        monitor!(mon, :warped, warped)
    end
    if haskey(mon, :warped0)
        warped = warp(moving0, ϕ)
        monitor!(mon, :warped0, warped)
    end
    mon
end

cudatype(::Type{T}) where {T<:Union{Float32,Float64}} = T
cudatype(::Any) = Float32

myconvert(::Type{Array{T}}, A::Array{T}) where {T} = A
myconvert(::Type{Array{T}}, A::AbstractArray) where {T} = copyto!(Array{T}(undef, size(A)), A)

end # module
