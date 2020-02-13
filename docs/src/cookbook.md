# [Stack-by-stack optimization with RegisterWorkerApertures: a cookbook](@id cookbook)

To learn how to use this package, we'll use a small 2d image sequence defined in BlockRegistration

```julia
using Distributed, StaticArrays, RegisterWorkerShell, JLD, FileIO, ImageView
wpids = addprocs(2)   # add two workers (you can use more if you have the hardware)
@everywhere using RegisterWorkerApertures, RegisterDriver

#### Load image on the master process
# Normally this might be `img = load("myimagefile")`, but this is a demo
using BlockRegistration                                # just needed for this demo
brdir = dirname(dirname(pathof(BlockRegistration)))    # directory of BlockRegistration
include(joinpath(brdir, "test", "gen2d.jl"));          # defines `img`

#### Choose the fixed image and set up the parameters (this is similar to BlockRegistration)
fixedidx = (nimages(img)+1) ÷ 2  # ÷ can be obtained with "\div[TAB]"
fixed = img[time=fixedidx]

# Important: you should manually inspect fixed to make sure there are
# no anomalies. Do not proceed to the next step until you have done this.

# Choose the maximum amount of movement you want to allow (set this by visual inspection)
mxshift = (30, 30)  # 30 pixels along each spatial axis for a 2d+time image
# Pick a grid size for your registration. Finer grids allow more
# "detail" in the deformation, but also have more parameters and
# therefore require higher SNR data.
gridsize = (3, 3)   # it's fine to choose something anisotropic

# Choose volume regularization penalty. See the help for BlockRegistration.
λ = 1e-5

# Compute the nodes from the image and grid
nodes = map(axes(fixed), gridsize) do ax, g
    range(first(ax), stop=last(ax), length=g)
end

#### Set up the workers, the monitor, and run it via the driver
# Create the worker algorithm structures. We assign one per worker process.
algorithm = [Apertures(fixed, nodes, mxshift, λ; pid=wpids[i], correctbias=false) for i = 1:length(wpids)]

# Set up the "monitor" which aggregates the results from the workers
mon = monitor(algorithm, (), Dict{Symbol,Any}(:u=>ArrayDecl(Array{SVector{2,Float64},2}, gridsize)))

# Load the appropriate mismatch package
mm_package_loader(algorithm)

# Define the output file and run the job
fileout = "results.register"
@time driver(fileout, algorithm, img, mon)

# Append important extra information to the file
jldopen(fileout, "r+") do io
    write(io, "fixedidx", fixedidx)
    write(io, "nodes", nodes)
end
```

You should see some output showing which workers are working on which time slices.

To visualize the results, let's load the deformation and apply it to the image sequence:

```julia
u = load(fileout, "u")
ϕs = griddeformations(u, nodes)   # defined in RegisterDeformation

imgw = similar(img, Gray{Float32});   # eltype needs to be able to store NaN
for i = 1:nimages(img)
    # Apply the deformation to the "recorded" image
    imgw[:,:,i] = warp(img[:,:,i], ϕs[i])
end
```

(See the documentation for BlockRegistration for how to do this when your images
are too large to hold in memory.)

Now you can visualize it with `imshow(imgw)`.

## Key differences from BlockRegistration

Most of the parameters and choices you'll make are similar to what you'd do if you're
using BlockRegistration directly.
The key changes are that instead of calling the various stages of mismatch calculation
and optimization, you instead set up the "algorithm" (one per worker), monitor, and
call the driver.

You can read the help for these in the API reference.
