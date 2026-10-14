#!/usr/bin/env bash
# Build mLibKriging (the Octave binding) from a libKriging checkout into OUT_DIR, which
# then goes on OCTAVE_PATH; install the Octave Jupyter kernel.
#   tools/install/octave.sh LIBKRIGING_SRC OUT_DIR
# Built from source for both channels (master: libKriging master; release: the last
# release tag): the published binaries target another Octave version than Ubuntu's.
# Needs octave, octave-dev, liblapack-dev, libopenblas-dev.
set -euo pipefail
src=$(cd "$1" && pwd); out=$(mkdir -p "$2" && cd "$2" && pwd)
if [ ! -f "$out/mLibKriging.mex" ]; then
  cmake -S "$src" -B "$src/build-octave" -DCMAKE_BUILD_TYPE=Release -DENABLE_OCTAVE_BINDING=on \
    -DENABLE_PYTHON_BINDING=off -DENABLE_MATLAB_BINDING=off -DBUILD_SHARED_LIBS=off -DSTATIC_LIB=on
  cmake --build "$src/build-octave" --target mLibKriging -j "$(nproc)"
  cp "$src/build-octave/bindings/Octave/mlibkriging/mLibKriging.mex" "$src"/bindings/Octave/mlibkriging/*.m "$out/"
fi
python -m pip install nbclient nbformat ipykernel octave_kernel
OCTAVE_PATH="$out" octave --no-gui --eval 'k = Kriging([0; 1; 0.5], [0; 1; 0.5], "gauss"); printf("mLibKriging OK (theta %g)\n", k.theta())'
