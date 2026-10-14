#!/usr/bin/env bash
# Install JLibKriging.jl (compiled against a libKriging checkout) and the Julia Jupyter
# kernel into the active Julia environment.
#   tools/install/julia.sh LIBKRIGING_SRC [KERNEL_NAME]   (default kernel name: julia-1.11,
#                                                          the one of the notebooks)
# JLibKriging.jl's build (deps/build.jl) compiles the checkout given by
# LIBKRIGING_SRC_DIR and generates the module from its bindings/Julia/jlibkriging.
# Already installed (e.g. a restored ~/.julia depot): only the kernel is (re)installed,
# and LIBKRIGING_SRC need not exist.
set -euo pipefail
kernel=${2:-julia-1.11}
python -m pip install nbclient nbformat ipykernel jupyter_client
if ! julia -e 'using JLibKriging, IJulia, DelimitedFiles' 2> /dev/null; then
  LIBKRIGING_SRC_DIR=$(cd "$1" && pwd); export LIBKRIGING_SRC_DIR
  julia -e 'using Pkg
            Pkg.add(url="https://github.com/libKriging/JLibKriging.jl")
            Pkg.add(["IJulia", "DelimitedFiles"])'
fi
julia -e "using IJulia; installkernel(\"Julia\"; specname=\"$kernel\")
          using JLibKriging; println(\"JLibKriging OK\")"
