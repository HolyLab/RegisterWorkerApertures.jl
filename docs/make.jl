using Documenter
using RegisterWorkerApertures

makedocs(
    sitename = "RegisterWorkerApertures",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [RegisterWorkerApertures],
    authors = "Timothy E. Holy",
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
