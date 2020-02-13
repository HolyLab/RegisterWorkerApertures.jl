# Notes on registering biological sequences

When registering image sequences such as those from calcium imaging, sometimes a number of pre-processing stages can be useful.

## Filtering & shot-noise transformation

[`PreprocessSNF`](https://holylab.github.io/RegisterCore.jl/dev/api/#RegisterCore.PreprocessSNF) can be useful to smooth, background-subtract,
bias-correct, and transform shot noise into an approximately stationary gaussian process.

Here's a demo:

```julia
# Make sure the pixelspacing property is set correctly; edit the .imagine
# file with a text editor if necessary or use the AxisArray as above.
ps = pixelspacing(img)
# Define preprocessing. Here we'll highpass filter over 25μm, but these
# numbers are likely to be image-dependent.
σ = 25μm
sigmahp = Float64[σ/x for x in ps]
sigmalp = zero(sigmahp)  # lowpass filtering is not currently recommended

# The pco cameras have a bias of 100 in "digital number"
# units. Convert this into the units of the image intensity.
# If you're using a different camera, or using a PMT, this won't apply to you.
# If you don't know anything better, you can set
#     bias = zero(eltype(img0))
bias = reinterpret(eltype(img), UInt16(100))
pp = PreprocessSNF(bias, sigmalp, sigmahp)
```  

If you're using such pre-processing, it's important to apply it consistently to both the fixed and moving images:

```julia
# To apply it to your fixed image:
fixed = pp(fixed)

# To pass it to the workers: use it as an argument to the algorithm
algorithm = [Apertures(fixed, nodes, mxshift, λ, pp; pid=wpids[i], correctbias=false) for i = 1:length(wpids)]
```

## Temporal median filtering

To reduce the impact of calcium "sparks," sometimes it is useful to apply temporal
median filtering to your images.
For this task, `mapwindow!` from ImageFiltering is recommended.
You can pre-allocate the output data file as a `mmapped` array:

```julia
using Mmap, ImageFiltering

# Specify the filtering window. This example is for an image sequence that
# has 3 spatial dimensions followed by one temporal dimension
const window = (1, 1, 1, 7)

open("my_filtered_data.dat", "w+") do io
    out = Mmap.mmap(io, Array{eltype(img),ndims(img)}, size(img))
    mapwindow!(median!, out, img, window)
end
```

`mapwindow!` also allows you to write out fewer time-slices than are present in the
original image; see the `indices` keyword.

## Artifact removal

Sometimes a very bright blob might float past your preparation.
Even if it doesn't enter
into the area where there are cells, it can bias the results.
One can set voxels of `fixed` outside the region of interest to `NaN`,
reducing the likelihood that such events become problematic.  

If the artifacts are much brighter than the tissue, one can also use `mappedarray` to
set them to `NaN` in the moving images:

```
const badthresh = 12345
imgthresh = mappedarray(x->x>badthresh ? nan(typeof(x)) : x, img)  # need ColorVectorSpace
```

For this to work you need to have your image encoded using a number type that supports
NaN, but you can also do this lazily with `mappedarray` (or use MappedArrays' `of_eltype`).

## Fixed image selection

If your experiment involves stimuli, it can often be best to select your fixed image
so that it does not include a strong response to stimuli.

This task is likely to be quite specific to individual users' experiments, so
no "canned" routines are provided.
HolyLab members performing olfaction experiments may want to use some of the
utilities in `LabShare/Jerry/juliafunc/Jerry_RegisterUtils`.

## Registering a subset of images

Typically the change of a tissue's shape is gradual. Routines here allow you to compute
the deformation for a subset of images and then use linear interpolation in the
time dimension to fill in intermediate time points.
See [tinterpolate](https://holylab.github.io/RegisterDeformation.jl/stable/api/#RegisterDeformation.tinterpolate).
