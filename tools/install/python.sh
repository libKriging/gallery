#!/usr/bin/env bash
# Install pylibkriging, then the notebooks' helpers (tools/lkbench) and runner.
#   tools/install/python.sh wheel   LIBKRIGING_SRC OUT_DIR   only build a wheel from a libKriging checkout
#   tools/install/python.sh master  WHEEL_DIR|LIBKRIGING_SRC install the wheel of WHEEL_DIR, or build it first
#   tools/install/python.sh release                          the last release, from PyPI
# The wheel needs BLAS/LAPACK headers (Ubuntu: liblapack-dev libopenblas-dev).
set -euo pipefail
tools=$(cd "$(dirname "$0")/.." && pwd)
build_wheel() {  # SRC OUT
  python -m pip install --upgrade pip
  python -m pip install -r "$1/bindings/Python/pylibkriging/requirements.txt" \
    -r "$1/bindings/Python/pylibkriging/dev-requirements.txt" cmake
  CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)} \
    python -m pip wheel --no-build-isolation --no-deps -w "$2" "$1"
}
case "$1" in
  wheel) build_wheel "$2" "$3"; exit 0 ;;
  master)
    if ! ls "$2"/pylibkriging-*.whl > /dev/null 2>&1; then
      build_wheel "$2" "$2/dist"; set -- "$1" "$2/dist"
    fi
    python -m pip install "$2"/pylibkriging-*.whl ;;
  release) python -m pip install pylibkriging ;;
  *) echo "usage: $0 wheel SRC OUT | master WHEEL_DIR|SRC | release" >&2; exit 2 ;;
esac
python -m pip install -e "${tools}[notebooks]"
python -c "import pylibkriging as lk; print('pylibkriging', lk.__version__, 'from', lk.__file__)"
