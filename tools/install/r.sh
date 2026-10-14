#!/usr/bin/env bash
# Install rlibkriging into the library directory LIB (added to R_LIBS_USER by the caller),
# and the R Jupyter kernel (IRkernel).
#   tools/install/r.sh master  LIB LIBKRIGING_SRC   build from a libKriging checkout (C++ core + bindings/R)
#   tools/install/r.sh release LIB                  the last release, from CRAN
# The master build follows libKriging's tools/r-linux-macos/build.sh: the core is compiled
# with R's own compilers as a static library, then bindings/R is built against it.
set -euo pipefail
channel=$1 lib=$(mkdir -p "$2" && cd "$2" && pwd)
export R_LIBS_USER="$lib"
if [ "$channel" = release ]; then
  Rscript -e 'install.packages("rlibkriging", lib = Sys.getenv("R_LIBS_USER"))'
else
  src=$(cd "$3" && pwd)
  # Rcpp & co. (bindings/R/rlibkriging/install/install.R would also install test-only packages)
  Rscript -e 'for (p in c("Rcpp", "RcppArmadillo", "RhpcBLASctl"))
                if (!requireNamespace(p, quietly = TRUE)) install.packages(p, lib = Sys.getenv("R_LIBS_USER"))'
  CC="$(R CMD config CC)" CXX="$(R CMD config CXX)" cmake -S "$src" -B "$src/build-r" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=off -DSTATIC_LIB=on \
    -DENABLE_OCTAVE_BINDING=off -DENABLE_MATLAB_BINDING=off -DENABLE_PYTHON_BINDING=off
  # install.lib also installs the bench/bench-* executables, which it does not build: build them
  # first (without the tests, which the full install target would also build)
  mapfile -t benches < <(cmake --build "$src/build-r" --target help | sed -n 's/^\.\.\. \(bench-[A-Za-z0-9_-]*\).*/\1/p')
  cmake --build "$src/build-r" --target "${benches[@]}" -j "$(nproc)"
  cmake --build "$src/build-r" --target install.lib -j "$(nproc)"
  export LIBKRIGING_PATH="$src/build-r/installed"
  cd "$src/bindings/R"
  mkdir -p Rlibs && touch Rlibs/.prepared   # dependencies already installed above
  R_LIBS="$lib" Rscript -e "Rcpp::compileAttributes(pkgdir = 'rlibkriging')"
  R_LIBS="$lib" MAKE_SHARED_LIBS=off STATIC_LIB=on MAKEFLAGS="-j$(nproc)" R CMD INSTALL --no-multiarch -l "$lib" rlibkriging
fi
python -m pip install nbclient nbformat ipykernel jupyter_client
Rscript -e 'if (!requireNamespace("IRkernel", quietly = TRUE)) install.packages("IRkernel", lib = Sys.getenv("R_LIBS_USER"))
            IRkernel::installspec(user = TRUE)
            cat("rlibkriging", as.character(packageVersion("rlibkriging")), "in", find.package("rlibkriging"), "\n")'
