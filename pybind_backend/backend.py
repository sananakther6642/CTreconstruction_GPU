"""
JIT-compiles and loads the pybind11 CT reconstruction backend.

Usage (from the repo root, or with the repo root on sys.path):

    from pybind_backend.backend import _backend, KERNEL_DIR
    volume = _backend.reconstruct_gpu_opt(proj, angles, ..., kernel_dir=KERNEL_DIR)

This mirrors the course's pybindextension/backend.py example, but links
the real reconstruction sources (src/utils.c, src/ct_cpu.c, src/ct_gpu.c)
instead of a single toy .cpp file.

Linux only. The C sources #include <hdf5.h> and <omp.h> unconditionally
(utils.c and ct_cpu.c respectively); neither HDF5 nor libomp is available
via a stock macOS toolchain (no Homebrew paths assumed here), and the
project's own Makefile has the same restriction. Bring up on the Linux
lab machines (pool15 / kale).

Build cache note: torch.utils.cpp_extension.load's cache key is the
literal flag strings and source contents -- it does NOT include CPU
architecture, hostname, or anything host-specific. If pool15 and kale
share a home directory (e.g. NFS), a .so built on one and imported on the
other would be silently reused rather than rebuilt. This is why the build
directory below is namespaced by hostname+machine: it forces a fresh
build (and thus a fresh binary matched to the actual CPU) on each
distinct host, even if they share $HOME. This is also what makes it
safe to compile with -march=native below: a native-arch binary is
never reused on a different host, so there is no cross-host SIGILL
risk from a mismatched instruction set.
"""
import os
import platform
import subprocess
import sys

# Match the Makefile's run-cpu target (OMP_NUM_THREADS/OMP_PROC_BIND=close/
# OMP_PLACES=cores). Without this, reconstruct_cpu's OpenMP threads are
# free to migrate across cores/sockets during a run -- observed on kale
# as per-epoch fp_cpu time climbing steadily (5.5s -> 14s+ over ~25
# epochs) instead of staying flat, consistent with threads drifting out
# of cache-friendly placement over time. Set only if the caller hasn't
# already exported these, so an explicit environment still wins.
#
# MUST run before `import torch` below, not after. Torch's own C++ init
# reads/caches OpenMP configuration once at import time; setting these
# afterward (even moments later) is too late -- GOMP's runtime, shared
# in-process with torch's, has already initialized against whatever the
# environment looked like when torch was imported. Measured on kale:
# with the env vars set after `import torch`, reconstruct_cpu ran ~13%
# slower than the CLI on both fp and bp (3.18s/1.11s vs 2.84s/0.94s),
# flat and reproducible -- not a bug in reconstruct_cpu itself (an
# identical in-process call with the vars set *before* import matched
# the CLI exactly, 2.84s/0.94s). Moving these four lines above the
# `torch.utils.cpp_extension` import closes the gap entirely.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")

from torch.utils.cpp_extension import load

if sys.platform == "darwin":
    raise RuntimeError(
        "pybind_backend requires Linux (HDF5 and libomp are not set up for "
        "macOS in this project). Build and run on pool15 or kale."
    )

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)
_src_dir = os.path.join(_repo_root, "src")
KERNEL_DIR = os.path.join(_repo_root, "kernels")

# Fail loudly and specifically at import time. Without this, a relocated
# pybind_backend/ or a missing kernel file surfaces as
# clCreateProgramWithSource failing inside gpu_init, which goes through
# CL_CHECK -> exit(1) -- no Python traceback, just a dead interpreter.
if not os.path.isdir(_src_dir):
    raise FileNotFoundError(
        f"expected C sources at {_src_dir} -- is pybind_backend/ still "
        f"directly under the repo root?"
    )

_REQUIRED_CL = [
    "bp_buffer.cl",
    "bp_buffer_opt.cl",
    "bp_image.cl",
    "fp_buffer.cl",
    "fp_image.cl",
]
_missing_cl = [f for f in _REQUIRED_CL if not os.path.isfile(os.path.join(KERNEL_DIR, f))]
if _missing_cl:
    raise FileNotFoundError(f"missing kernel files in {KERNEL_DIR}: {_missing_cl}")


def _hdf5_lib_flag():
    """Mirror the Makefile's ldconfig probe: Debian/Ubuntu package HDF5 as
    libhdf5_serial, other distros as libhdf5."""
    try:
        out = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, check=False
        ).stdout
        return "-lhdf5_serial" if "libhdf5_serial" in out else "-lhdf5"
    except Exception:
        return "-lhdf5"


def _hdf5_include_dirs():
    """Mirror the Makefile: /usr/include/hdf5/serial if present, else
    the plain system include dir."""
    serial_dir = "/usr/include/hdf5/serial"
    return [serial_dir] if os.path.isdir(serial_dir) else ["/usr/include"]


def _build_directory():
    """Namespace the JIT build dir by host so a shared home directory
    across lab machines can't hand one machine's compiled .so to
    another (see module docstring)."""
    tag = f"{platform.node()}-{platform.machine()}".replace("/", "_")
    build_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions", f"ct_recon-{tag}"
    )
    os.makedirs(build_dir, exist_ok=True)
    return build_dir


_backend = load(
    name="ct_recon",
    sources=[
        os.path.join(_here, "src", "ct_recon_bindings.cpp"),
        os.path.join(_src_dir, "utils.c"),
        os.path.join(_src_dir, "ct_cpu.c"),
        os.path.join(_src_dir, "ct_gpu.c"),
    ],
    extra_include_paths=[_src_dir] + _hdf5_include_dirs(),
    # -march=native + -ffast-math match the Makefile's CLI build exactly.
    # Originally left out here over a SIGILL worry (a native-arch binary
    # built on one host silently reused on another with a different
    # CPU) -- but _build_directory() below already namespaces the JIT
    # cache per-host, so that binary is never shared across machines and
    # the worry doesn't apply. Measured on kale: omitting -march=native
    # cost reconstruct_cpu a real ~1.8x slowdown (7.2s/epoch vs the
    # CLI's 3.9s/epoch, same OMP_NUM_THREADS/PROC_BIND/PLACES on both
    # sides) -- generic -O3 doesn't emit the AVX2 vectorization
    # -march=native does for fp_cpu's trilinear interpolation. Keeping
    # -ffast-math too since the CLI's own validated results already use
    # it; its process-wide FTZ/DAZ side effect on torch/numpy denormal
    # handling is a known, accepted tradeoff, not a new one. -fopenmp is
    # required in both cflags and ldflags or reconstruct_cpu's #pragma
    # omp loops silently run single-threaded.
    extra_cflags=["-O3", "-march=native", "-fopenmp", "-ffast-math",
                  "-DCL_TARGET_OPENCL_VERSION=120"],
    extra_ldflags=["-lOpenCL", _hdf5_lib_flag(), "-lm", "-fopenmp"],
    build_directory=_build_directory(),
)

__all__ = ["_backend", "KERNEL_DIR"]
