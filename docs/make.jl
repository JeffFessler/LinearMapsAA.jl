# push!(LOAD_PATH,"../src/")

using LinearMapsAA
using Documenter

DocMeta.setdocmeta!(LinearMapsAA, :DocTestSetup, :(using LinearMapsAA); recursive=true)

makedocs(;
    modules = [LinearMapsAA],
    authors = "Jeff Fessler <fessler@umich.edu> and contributors",
    repo = "https://github.com/JeffFessler/LinearMapsAA.jl/blob/{commit}{path}#{line}",
    sitename = "LinearMapsAA.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
#       canonical = "https://JeffFessler.github.io/LinearMapsAA.jl/stable",
#       assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/JeffFessler/LinearMapsAA.jl.git",
    devbranch = "main",
    devurl = "dev",
    versions = ["stable" => "v^", "dev" => "dev"]
#   push_preview = true,
)
