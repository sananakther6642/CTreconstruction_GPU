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
distinct host, even if they share $HOME.
"""
import os
import platform
import subprocess
import sys

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
    # No -march=native: the build cache is host-agnostic (see module
    # docstring), so a native-arch binary built on one machine could be
    # silently reused on another and crash with SIGILL. No -ffast-math
    # in this build: gcc links crtfastmath.o for it, which sets FTZ/DAZ
    # in MXCSR for the WHOLE PROCESS -- that would alter denormal
    # handling in torch/numpy for the rest of the interpreter, not just
    # this module. -fopenmp is required in both cflags and ldflags or
    # reconstruct_cpu's #pragma omp loops silently run single-threaded.
    extra_cflags=["-O3", "-fopenmp", "-DCL_TARGET_OPENCL_VERSION=120"],
    extra_ldflags=["-lOpenCL", _hdf5_lib_flag(), "-lm", "-fopenmp"],
    build_directory=_build_directory(),
)

__all__ = ["_backend", "KERNEL_DIR"]
