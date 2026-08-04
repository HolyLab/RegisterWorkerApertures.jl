"""
    RegisterWorkerApertures

Apertured (blocked) image registration worker for the Register ecosystem.

Divides the image domain into a grid of overlapping apertures, computes
per-aperture mismatch arrays, fits quadratic surfaces to those mismatches,
and solves a penalized optimization to recover a smooth deformation field.
Both CPU and CUDA execution paths are supported.

# Primary interface

- [`Apertures`](@ref) — construct a registration worker
- `monitor`, `monitor!` — re-exported from `RegisterWorkerShell`; create and
  update a dict that collects registration outputs across frames
- [`worker`](@ref) — execute one registration step (typically called by `driver`)
"""
module RegisterWorkerApertures

using CoordinateTransformations: CoordinateTransformations
using ImageCore: ImageCore, coords_spatial, nimages
using ImageTransformations: warp
using Interpolations: Interpolations
using RegisterCore: RegisterCore, maxshift
using RegisterDeformation: RegisterDeformation
using RegisterFit: RegisterFit, qfit
using RegisterMismatchCommon: RegisterMismatchCommon, allocate_mmarrays,
                              aperture_grid, correctbias, correctbias!,
                              default_aperture_width, mismatch,
                              mismatch_apertures
using RegisterOptimize: RegisterOptimize
using RegisterPenalty: RegisterPenalty, AffinePenalty, interpolate_mm!
using RegisterWorkerShell: RegisterWorkerShell, AbstractWorker, getindex_t,
                           monitor, monitor!
using SharedArrays: SharedArrays, sdata
# Note: RegisterMismatch/RegisterMismatchCuda is selected below
import RegisterWorkerShell: worker, init!, close!, load_mm_package, workertid

export Apertures, monitor, monitor!, worker

struct Apertures{A <: AbstractArray, K, N} <: AbstractWorker
    fixed::A
    nodes::NTuple{N, K}
    maxshift::NTuple{N, Int}
    affinepenalty::AffinePenalty{Float64, N}
    overlap::NTuple{N, Int}
    λrange::Union{Float64, Tuple{Float64, Float64}}
    thresh::Float64  # Ipopt requires Float64
    preprocess  # likely of type PreprocessSNF, but could be a function
    normalization::Symbol
    correctbias::Bool
    tid::Int
    dev::Int
    cuda_objects::Dict{Symbol, Any}
end

"""
    load_mm_package(dev)

Load the appropriate mismatch computation package.

Loads `RegisterMismatchCuda` (and `CUDA`) when `dev >= 0` (a CUDA device index),
or `RegisterMismatch` when `dev < 0` (CPU mode). Returns `nothing`.
"""
function load_mm_package(dev)
    if dev >= 0
        eval(:(using CUDA, RegisterMismatchCuda))
    else
        eval(:(using RegisterMismatch))
    end
    return nothing
end

"""
    init!(algorithm::Apertures)

Initialize the worker before registration begins.

When `algorithm.dev >= 0` (GPU mode), allocates device arrays and prepares
`CMStorage` via [`cuda_init!`](@ref). No-op in CPU mode.
"""
function init!(algorithm::Apertures)
    if algorithm.dev >= 0
        cuda_init!(algorithm)
    end
    return nothing
end

"""
    cuda_init!(algorithm)

Allocate CUDA resources for `algorithm` and store them in `algorithm.cuda_objects`.

Selects the device `algorithm.dev`, saves the previously active CUDA context, and
pre-allocates three objects:
- `:d_fixed` — `CuArray` copy of `algorithm.fixed`
- `:d_moving` — same-sized scratch buffer
- `:cms` — `CMStorage` sized to the aperture grid and `algorithm.maxshift`
"""
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
    T = cuda_eltype(eltype(fixed))
    d_fixed = CuArray{T}(sdata(fixed))
    algorithm.cuda_objects[:d_fixed] = d_fixed
    algorithm.cuda_objects[:d_moving] = similar(d_fixed)
    gridsize = map(length, algorithm.nodes)
    aperture_width = default_aperture_width(algorithm.fixed, gridsize)
    return algorithm.cuda_objects[:cms] = CMStorage{T}(undef, aperture_width, algorithm.maxshift)
end

"""
    close!(algorithm::Apertures)

Restore the CUDA context that was active before [`init!`](@ref) was called.

When `algorithm.dev >= 0` (GPU mode) and a prior context was saved during
`init!`, reactivates that context. No-op in CPU mode.
"""
function close!(algorithm::Apertures)
    if algorithm.dev >= 0
        if old_active_context != nothing
            activate(old_active_context)
        end
    end
    return nothing
end

"""
    Apertures(fixed, nodes, maxshift, λrange, [preprocess=identity]; kwargs...)

Create a worker object for apertured (blocked) image registration.

`fixed` is the single reference image. `nodes` is an `N`-tuple of node vectors
(typically `range` objects) that define the aperture grid. `maxshift` is an
`N`-tuple of integers giving the maximum shift (in pixels) to evaluate in each
dimension. `λrange` is either a single `Float64` (fixed regularization
coefficient) or a `(λmin, λmax)` tuple; in the latter case the best `λ` is
selected automatically via `auto_λ` and the search intermediates are available
via `monitor`. `preprocess` is an optional function applied to each moving image
before registration; `fixed` should already have the same transformation applied
(see also `PreprocessSNF`).

# Keyword arguments

- `overlap`: `N`-vector of integers specifying the overlap (in pixels) between
  adjacent apertures; default `zeros(Int, N)`.
- `normalization`: `:pixels` (default) normalizes mismatch by aperture area;
  `:intensity` normalizes by total intensity in the aperture.
- `thresh_fac`: factor used to compute `thresh` when not supplied explicitly;
  default `(0.5)^ndims(fixed)`.
- `thresh`: mismatch threshold below which a quadratic fit is attempted;
  computed from `thresh_fac` and `normalization` by default.
- `correctbias`: apply bias correction to mismatch arrays; default `true`.
- `tid`: thread/process ID for this worker; default `1`.
- `dev`: CUDA device index (`>= 0` to use GPU, `-1` for CPU); default `-1`.

Pass the returned object to `driver` or directly to `worker`.

# Example

```jldoctest
julia> using RegisterWorkerApertures

julia> fixed = rand(Float64, 50, 50);

julia> nodes = (range(1, 50, length=5), range(1, 50, length=7));

julia> alg = Apertures(fixed, nodes, (5, 5), (1e-6, 100.0));

julia> alg.dev
-1
```
"""
function Apertures(fixed, nodes::NTuple{N, K}, maxshift, λrange, preprocess = identity; overlap = zeros(Int, N), normalization = :pixels, thresh_fac = (0.5)^ndims(fixed), thresh = nothing, correctbias::Bool = true, tid = 1, dev = -1) where {K, N}
    gridsize = map(length, nodes)
    overlap_t = (overlap...,) #Make tuple
    length(overlap) == N || throw(DimensionMismatch("overlap must have $N entries"))
    nimages(fixed) == 1 || error("Register to a single image")
    isa(λrange, Number) || isa(λrange, Tuple{Number, Number}) || error("λrange must be a number or 2-tuple")
    normalization ∈ (:pixels, :intensity) ||
        error("normalization must be :pixels or :intensity, got :$normalization")
    if thresh == nothing
        # Match the mismatch denominator the apertures are scored against, so
        # `thresh_fac` means the same fraction under either normalization.
        thresh = (thresh_fac / prod(gridsize)) *
            (normalization == :pixels ? length(fixed) : sum(abs2, fixed))
    end
    λrange = isa(λrange, Number) ? Float64(λrange) : (Float64(first(λrange)), Float64(last(λrange)))
    return Apertures{typeof(fixed), K, N}(fixed, nodes, maxshift, AffinePenalty{Float64, N}(nodes, first(λrange)), overlap_t, λrange, Float64(thresh), preprocess, normalization, correctbias, tid, dev, Dict{Symbol, Any}())
end

"""
    worker(algorithm::Apertures, img, tindex, mon)

Perform apertured registration of the image at time index `tindex` in `img`
against the fixed reference stored in `algorithm`.

`img` is an image array with a time axis; `tindex` is the integer index into
that axis. `mon` is a monitoring dict created by `monitor`; it is updated
in-place with any requested outputs and returned.

Monitored keys written unconditionally:
- `:mismatch` — total mismatch value after optimization
- `:u` — the deformation field `ϕ.u`

Monitored keys written only when `λrange` is a `(λmin, λmax)` tuple:
- `:λ` — chosen regularization coefficient
- `:λs` — all `λ` values tried by `auto_λ`
- `:datapenalty` — data penalty at each `λ`
- `:sigmoid_quality` — quality of the sigmoid fit

Monitored keys written only when present in `mon`:
- `:warped` — preprocessed moving image warped by the recovered deformation `ϕ`
- `:warped0` — raw (unpreprocessed) moving image warped by `ϕ`
"""
function worker(algorithm::Apertures, img, tindex, mon)
    moving0 = getindex_t(img, tindex)
    moving = algorithm.preprocess(moving0)
    gridsize = map(length, algorithm.nodes)
    use_cuda = algorithm.dev >= 0
    if use_cuda
        device!(CuDevice(algorithm.dev))
        d_fixed = algorithm.cuda_objects[:d_fixed]
        d_moving = algorithm.cuda_objects[:d_moving]
        cms = algorithm.cuda_objects[:cms]
        copyto!(d_moving, moving)
        cs = coords_spatial(img)
        aperture_centers = aperture_grid(map(d -> size(img, d), cs), gridsize)
        mms = allocate_mmarrays(eltype(cms), gridsize, algorithm.maxshift)
        mismatch_apertures!(mms, d_fixed, d_moving, aperture_centers, cms; normalization = algorithm.normalization)
    else
        #mms = mismatch_apertures(algorithm.fixed, moving, gridsize, algorithm.maxshift; normalization=algorithm.normalization)
        cs = coords_spatial(img) #
        aperture_centers = aperture_grid(map(d -> size(img, d), cs), gridsize)
        aperture_width = default_aperture_width(algorithm.fixed, gridsize, algorithm.overlap)  #
        mms = mismatch_apertures(algorithm.fixed, moving, aperture_centers, aperture_width, algorithm.maxshift; normalization = algorithm.normalization)  #
    end
    # displaymismatch(mms, thresh=10)
    if algorithm.correctbias
        correctbias!(mms)
    end
    E0 = zeros(size(mms))
    cs = Array{Any}(undef, size(mms))
    Qs = Array{Any}(undef, size(mms))
    thresh = algorithm.thresh
    for i in 1:length(mms)
        E0[i], cs[i], Qs[i] = qfit(mms[i], thresh; opt = false)
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
    return mon
end

"""
    cuda_eltype(T)

Return the element type to use for CUDA arrays given array element type `T`.

Passes `Float32` and `Float64` through unchanged; maps all other types to `Float32`.
"""
cuda_eltype(::Type{T}) where {T <: Union{Float32, Float64}} = T
cuda_eltype(::Any) = Float32

workertid(alg::Apertures) = alg.tid

end # module
