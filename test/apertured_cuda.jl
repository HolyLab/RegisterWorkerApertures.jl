### Apertured registration
# Create the data
img = testimage("cameraman")
gridsizeg = (4,4) # for generation
shift_amplitude = 10
u_dfm = shift_amplitude*randn(2, gridsizeg...)
nodesg = map(d->range(1, stop=size(img,d), length=gridsizeg[d]), (1,2))
ϕ_dfm = GridDeformation(u_dfm, nodesg)
wimg = warp(img, ϕ_dfm)
o = 3*shift_amplitude
fixed = img[o+1:size(img,1)-o, o+1:size(img,2)-o]
moving = wimg[o+1:size(img,1)-o, o+1:size(img,2)-o]

# Set up the range of λ, and prepare for plotting
λrange = (1e-6,10)

# To make sure it runs, try the example in the docs, even though it's
# not well-tuned for this case
pp = img -> imfilter(img, KernelFactors.IIRGaussian((3,3)))
nodes = (range(1, stop=size(fixed,1), length=5), range(1, stop=size(fixed,2), length=7))
fixedfilt = pp(fixed)
maxshift = (30,30)
alg = Apertures(fixedfilt, nodes, maxshift, λrange, pp, dev=0)
mm_package_loader(alg)
mon = monitor(alg, (), Dict(:λs=>0, :datapenalty=>0, :λ=>0, :u=>0, :warped0 => Array{Float64}(undef, size(fixed))))
mon = driver(alg, moving, mon)
datapenalty = mon[:datapenalty]
@test !all(mon[:warped0] .== 0)
# plot(x=λs, y=datapenalty, xintercept=[mon[:λ]], Geom.point, Geom.vline, Guide.xlabel("λ"), Guide.ylabel("Data penalty"), Scale.x_log10)

# Perform the registration
gridsize = (17,17)  # for correction
nodes = map(d->range(1, stop=size(fixed,d), length=gridsize[d]), (1,2))
umax = maximum(abs.(u_dfm))
maxshift = (ceil(Int, umax)+5, ceil(Int, umax)+5)
algorithm = RegisterWorkerApertures.Apertures(fixed, nodes, maxshift, λrange, dev=0)
mm_package_loader(algorithm)
mon = Dict{Symbol,Any}(:u => Array{SVector{2,Float64}}(undef, gridsize),
                       :mismatch => 0.0,
                       :λ => 0.0,
                       :datapenalty => 0,
                       :sigmoid_quality => 0.0,
                       :warped => copy(moving))
mon = driver(algorithm, moving, mon)

# With aperture overlap
apertureoverlap = 0.3;  #Aperture overlap percentage (between 0 and 1)
aperture_width = default_aperture_width(fixed, gridsize)
overlap_t = map(x->round(Int64,x*apertureoverlap), aperture_width)
algorithm = RegisterWorkerApertures.Apertures(fixed, nodes, maxshift, λrange; overlap=overlap_t, dev=0)
mm_package_loader(algorithm)
mon_overlap = Dict{Symbol,Any}(:u => Array{SVector{2,Float64}}(undef, gridsize),
                       :mismatch => 0.0,
                       :λ => 0.0,
                       :datapenalty => 0,
                       :sigmoid_quality => 0.0,
                       :warped => copy(moving))
mon_overlap = driver(algorithm, moving, mon_overlap)

# Analysis
ϕ = GridDeformation(mon[:u], nodes)
ϕ_overlap = GridDeformation(mon_overlap[:u], nodes)

gd0 = warpgrid(ϕ_dfm, showidentity=true)
ϕi = interpolate(ϕ_dfm)

gd1 = warpgrid(ϕi(interpolate(ϕ)), showidentity=true)
gd1_overlap = warpgrid(ϕi(interpolate(ϕ_overlap)), showidentity=true)

r0 = ratio(mismatch0(fixed, moving), 0)
r1 = ratio(mismatch0(fixed, mon[:warped]), 0)
r1_overlap = ratio(mismatch0(fixed, mon_overlap[:warped]), 0)
@test r1 < r0
@test r1_overlap < r0

# Consider:
# using RegisterGUI
# ImagePlayer.view(gd0)
# ImagePlayer.view(gd1)
# showoverlay(fixed, moving)
# showoverlay(fixed, mon[:warped])
#
# plot(x=λs, y=mon[:datapenalty], xintercept=[mon[:λ]], Geom.point, Geom.vline, Guide.xlabel("λ"), Guide.ylabel("Data penalty"), Scale.x_log10)

nothing
