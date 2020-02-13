# RegisterWorkerApertures.jl

This package provides convenient distributed computing support for the
[BlockRegistration]
suite of image registration packages.
The overall task is organized by a "driver" process ([RegisterDriver](https://github.com/HolyLab/RegisterDriver.jl)),
which assigns individual images (which may be 2d or 3d) to
"workers" which perform the computations of registration.

Both CPU and GPU computing are supported; for GPU computing you should
have one device (one GPU card) per worker.

If your images are not large, you may find it easier to use [BlockRegistration](https://github.com/HolyLab/BlockRegistration.jl) directly.
It's recommended that you read the documentation of that package before starting with this one.
