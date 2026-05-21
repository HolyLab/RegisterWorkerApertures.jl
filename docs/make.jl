using Documenter
using RegisterWorkerApertures

makedocs(
    sitename = "RegisterWorkerApertures",
    format = Documenter.HTML(),
    modules = [RegisterWorkerApertures],
    authors = "Timothy E. Holy",
    checkdocs = :exports,
    linkcheck = !("skiplinks" in ARGS),
    pages = [
        "Home" => "index.md",
        "cookbook.md",
        "calcium_imaging.md",
        "api.md"
    ],
)

deploydocs(
    repo = "github.com/HolyLab/RegisterWorkerApertures.jl.git",
)
