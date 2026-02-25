using MultiDeviceLinearAlgebra
using Documenter

DocMeta.setdocmeta!(MultiDeviceLinearAlgebra, :DocTestSetup, :(using MultiDeviceLinearAlgebra); recursive=true)

makedocs(;
    modules=[MultiDeviceLinearAlgebra],
    authors="Kyle Beggs (beggskw@gmail.com) and contributors",
    sitename="MultiDeviceLinearAlgebra.jl",
    format=Documenter.HTML(;
        canonical="https://kylebeggs.github.io/MultiDeviceLinearAlgebra.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kylebeggs/MultiDeviceLinearAlgebra.jl",
    devbranch="main",
)
