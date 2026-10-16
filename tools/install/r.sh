#!/usr/bin/env bash
# Install rlibkriging into the library directory LIB (added to R_LIBS_USER by the caller),
# and the R Jupyter kernel (IRkernel).
#   tools/install/r.sh master  LIB LIBKRIGING_SRC   build from a libKriging checkout (C++ core + bindings/R)
#   tools/install/r.sh release LIB                  the last release, from CRAN
#   tools/install/r.sh kernel  LIB                  only the R Jupyter kernel (rlibkriging already in LIB)
# The master build follows libKriging's tools/r-linux-macos/build.sh: the core is compiled
# with R's own compilers as a static library, then bindings/R is built against it.
set -euo pipefail
channel=$1 lib=$(mkdir -p "$2" && cd "$2" && pwd)
export R_LIBS_USER="$lib"
if [ "$channel" = kernel ]; then
  :
elif [ "$channel" = release ]; then
  Rscript -e 'install.packages("rlibkriging", lib = Sys.getenv("R_LIBS_USER"))'
else
  src=$(cd "$3" && pwd)
  # Rcpp & co. (bindings/R/rlibkriging/install/install.R would also install test-only packages)
  Rscript -e 'for (p in c("Rcpp", "RcppArmadillo", "RhpcBLASctl"))
                if (!requireNamespace(p, quietly = TRUE)) install.packages(p, lib = Sys.getenv("R_LIBS_USER"))'
  CC="$(R CMD config CC)" CXX="$(R CMD config CXX)" cmake -S "$src" -B "$src/build-r" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=off -DSTATIC_LIB=on \
    -DENABLE_OCTAVE_BINDING=off -DENABLE_MATLAB_BINDING=off -DENABLE_PYTHON_BINDING=off
  cmake --build "$src/build-r" --target install.lib -j "$(nproc)"
  export LIBKRIGING_PATH="$src/build-r/installed"
  pushd "$src/bindings/R" > /dev/null
  mkdir -p Rlibs && touch Rlibs/.prepared   # dependencies already installed above
  R_LIBS="$lib" Rscript -e "Rcpp::compileAttributes(pkgdir = 'rlibkriging')"
  R_LIBS="$lib" MAKE_SHARED_LIBS=off STATIC_LIB=on MAKEFLAGS="-j$(nproc)" R CMD INSTALL --no-multiarch -l "$lib" rlibkriging
  popd > /dev/null
fi
python -m pip install nbclient nbformat ipykernel jupyter_client
# The kernelspec is written directly, as IRkernel::installspec() would, without its check of
# the `jupyter kernelspec` command (which fails on CI runners).
# shellcheck disable=SC2016  # R code, not shell expansions
Rscript -e 'if (!requireNamespace("IRkernel", quietly = TRUE)) install.packages("IRkernel", lib = Sys.getenv("R_LIBS_USER"))
            dest <- file.path(Sys.getenv("JUPYTER_DATA_DIR", file.path(Sys.getenv("HOME"), ".local/share/jupyter")), "kernels", "ir")
            dir.create(dest, recursive = TRUE, showWarnings = FALSE)
            spec <- system.file("kernelspec", package = "IRkernel")
            file.copy(list.files(spec, full.names = TRUE), dest, overwrite = TRUE)
            k <- jsonlite::fromJSON(file.path(dest, "kernel.json"))
            k$argv[1] <- file.path(R.home("bin"), "R")
            jsonlite::write_json(k, file.path(dest, "kernel.json"), auto_unbox = TRUE, pretty = TRUE)
            cat("R kernel in", dest, "\n")
            cat("rlibkriging", as.character(packageVersion("rlibkriging")), "in", find.package("rlibkriging"), "\n")'
