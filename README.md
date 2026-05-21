# RegisterWorkerApertures.jl

[![CI](https://github.com/HolyLab/RegisterWorkerApertures.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/RegisterWorkerApertures.jl/actions/workflows/CI.yml)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/RegisterWorkerApertures.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/RegisterWorkerApertures.jl/dev)
[![codecov](https://codecov.io/gh/HolyLab/RegisterWorkerApertures.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/RegisterWorkerApertures.jl)
[![version](https://juliahub.com/ui/Packages/General/RegisterWorkerApertures/badge.svg)](https://juliahub.com/ui/Packages/General/RegisterWorkerApertures)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This package supports distributed computing to accelerate image registration.
It wraps the [BlockRegistration](https://github.com/HolyLab/BlockRegistration.jl) framework.

For an introduction, see the documentation.

## Installation

This package is registered in the [HolyLab registry](https://github.com/HolyLab/HolyLabRegistry).
Add that registry once, then install normally:

```julia
using Pkg
pkg"registry add General https://github.com/HolyLab/HolyLabRegistry.git"
Pkg.add("RegisterWorkerApertures")
```
